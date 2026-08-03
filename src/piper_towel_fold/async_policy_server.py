"""Piper wrapper around LeRobot async PolicyServer with pi05 memory optimizations."""

from __future__ import annotations

import logging
import pickle  # nosec
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from typing import Any

import grpc

from .async_features import observation_features_from_dataset
from .async_image_codec import (
    decompress_observation_images,
    observation_has_jpeg_payloads,
)
from .offline_infer import load_policy


def _import_policy_server():
    try:
        from lerobot.async_inference.configs import PolicyServerConfig
        from lerobot.async_inference.policy_server import PolicyServer
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.async_inference.helpers import get_logger
    except ImportError as exc:
        raise ImportError(
            "LeRobot async inference is not installed. "
            'Install with: pip install "lerobot[async,pi]"'
        ) from exc

    return PolicyServer, PolicyServerConfig, get_logger, services_pb2, services_pb2_grpc


class PiperPolicyServer:
    """PolicyServer with load_policy() overrides for low-VRAM pi05 inference."""

    def __init__(
        self,
        config: Any,
        *,
        inference_dtype: str | None = None,
        compile_model: bool | None = None,
        num_inference_steps: int | None = None,
    ) -> None:
        (
            base_server_cls,
            _,
            get_logger,
            services_pb2,
            _,
        ) = _import_policy_server()

        self._base_server_cls = base_server_cls
        self._services_pb2 = services_pb2
        self.logger = get_logger("policy_server")

        self.config = config
        self.inference_dtype = inference_dtype
        self.compile_model = compile_model
        self.num_inference_steps = num_inference_steps

        self._server = base_server_cls(config)
        self._policy_preloaded = False
        self._preloaded_policy_path: str | None = None
        self._patch_send_policy_instructions()
        self._patch_send_observations()

    def _apply_policy_specs(
        self,
        policy_specs: Any,
        *,
        log_client: str | None = None,
    ) -> float:
        from lerobot.async_inference.constants import SUPPORTED_POLICIES

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        if log_client is not None:
            self.logger.info(
                f"Receiving policy instructions from {log_client} | "
                f"Policy type: {policy_specs.policy_type} | "
                f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
                f"Actions per chunk: {policy_specs.actions_per_chunk} | "
                f"Device: {policy_specs.device}"
            )

        self._server.device = policy_specs.device
        self._server.policy_type = policy_specs.policy_type
        self._server.lerobot_features = policy_specs.lerobot_features
        self._server.actions_per_chunk = policy_specs.actions_per_chunk

        if (
            self._policy_preloaded
            and self._server.policy is not None
            and self._preloaded_policy_path == policy_specs.pretrained_name_or_path
        ):
            self.logger.info(
                "Policy already loaded at server startup; skipping model reload."
            )
            return 0.0

        start = time.perf_counter()
        policy_config, policy, make_pre_post_processors = load_policy(
            policy_specs.pretrained_name_or_path,
            policy_specs.device,
            inference_dtype=self.inference_dtype,
            compile_model=self.compile_model,
            num_inference_steps=self.num_inference_steps,
        )
        self._server.policy = policy

        device_override = {"device": policy_specs.device}
        self._server.preprocessor, self._server.postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=policy_config.pretrained_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        elapsed = time.perf_counter() - start
        self._policy_preloaded = True
        self._preloaded_policy_path = policy_specs.pretrained_name_or_path
        self.logger.info(f"Time taken to put policy on {policy_specs.device}: {elapsed:.4f}s")
        return elapsed

    def preload_policy(
        self,
        *,
        policy_type: str,
        policy_path: str,
        dataset_root: str,
        device: str,
        actions_per_chunk: int,
    ) -> None:
        from lerobot.async_inference.helpers import RemotePolicyConfig

        self.logger.info(
            f"Preloading policy at startup | type={policy_type} | path={policy_path} | device={device}"
        )
        policy_specs = RemotePolicyConfig(
            policy_type,
            policy_path,
            observation_features_from_dataset(dataset_root),
            actions_per_chunk,
            device,
        )
        elapsed = self._apply_policy_specs(policy_specs)
        if elapsed > 0:
            self.logger.info(f"Startup preload finished in {elapsed:.1f}s")
        else:
            self.logger.info("Startup preload finished (policy was already loaded)")

    def _patch_send_policy_instructions(self) -> None:
        original = self._server.SendPolicyInstructions

        def send_policy_instructions(request, context):  # noqa: N802
            if not self._server.running:
                self.logger.warning("Server is not running. Ignoring policy instructions.")
                return self._services_pb2.Empty()

            client_id = context.peer()
            policy_specs = pickle.loads(request.data)  # nosec

            try:
                from lerobot.async_inference.helpers import RemotePolicyConfig
            except ImportError as exc:
                raise ImportError(
                    'LeRobot async inference is not installed. '
                    'Install with: pip install "lerobot[async,pi]"'
                ) from exc

            if not isinstance(policy_specs, RemotePolicyConfig):
                raise TypeError(
                    f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}"
                )

            self._apply_policy_specs(policy_specs, log_client=client_id)
            return self._services_pb2.Empty()

        self._server.SendPolicyInstructions = send_policy_instructions
        _ = original

    def _patch_send_observations(self) -> None:
        """Decode Piper JPEG image payloads before LeRobot enqueue/inference."""
        original = self._server.SendObservations

        def send_observations(request_iterator, context):  # noqa: N802
            from lerobot.async_inference.helpers import TimedObservation
            from lerobot.transport.utils import receive_bytes_in_chunks

            if not self._server.running:
                self.logger.warning("Server is not running. Ignoring observations.")
                return self._services_pb2.Empty()

            client_id = context.peer()
            self.logger.debug(f"Receiving observations from {client_id}")

            receive_time = time.time()
            start_deserialize = time.perf_counter()
            received_bytes = receive_bytes_in_chunks(
                request_iterator,
                None,
                self._server.shutdown_event,
                self.logger,
            )
            timed_observation = pickle.loads(received_bytes)  # nosec
            deserialize_time = time.perf_counter() - start_deserialize

            if not isinstance(timed_observation, TimedObservation):
                raise TypeError(
                    f"Expected TimedObservation, got {type(timed_observation)}"
                )

            raw_observation = timed_observation.get_observation()
            if observation_has_jpeg_payloads(raw_observation):
                decode_started = time.perf_counter()
                timed_observation.observation = decompress_observation_images(raw_observation)
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
                self.logger.debug(
                    f"Decoded JPEG observation #{timed_observation.get_timestep()} "
                    f"in {decode_ms:.1f}ms | wire={len(received_bytes) / (1024 * 1024):.2f}MB"
                )

            obs_timestep = timed_observation.get_timestep()
            obs_timestamp = timed_observation.get_timestamp()
            fps_metrics = self._server.fps_tracker.calculate_fps_metrics(obs_timestamp)

            self.logger.debug(
                f"Received observation #{obs_timestep} | "
                f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                f"Target: {fps_metrics['target_fps']:.2f} | "
                f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
            )
            self.logger.debug(
                f"Server timestamp: {receive_time:.6f} | "
                f"Client timestamp: {obs_timestamp:.6f} | "
                f"Deserialization time: {deserialize_time:.6f}s"
            )

            if not self._server._enqueue_observation(timed_observation):
                self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

            return self._services_pb2.Empty()

        self._server.SendObservations = send_observations
        _ = original

    @property
    def servicer(self) -> Any:
        return self._server

    def stop(self) -> None:
        self._server.stop()


def serve_piper_policy_server(
    config: Any,
    *,
    inference_dtype: str | None = None,
    compile_model: bool | None = None,
    num_inference_steps: int | None = None,
    preload_policy: dict[str, Any] | None = None,
) -> None:
    _, _, _, _, services_pb2_grpc = _import_policy_server()

    logging.info(pformat(asdict(config)))
    policy_server = PiperPolicyServer(
        config,
        inference_dtype=inference_dtype,
        compile_model=compile_model,
        num_inference_steps=num_inference_steps,
    )

    if preload_policy is not None:
        policy_server.preload_policy(**preload_policy)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server.servicer, server)
    server.add_insecure_port(f"{config.host}:{config.port}")

    policy_server.logger.info(f"PolicyServer started on {config.host}:{config.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")
