# HTTP Control Server Design

## Goal

Provide a headless HTTP service on the robot IPC that starts and stops the
remote Policy Server and the local Policy Client workflow. The service uses
`configs/fold_cloth.json`, listens on port `12123` by default, and keeps a
separate log for each managed process.

The HTTP implementation is independent of the `piper_train` package. It lives
under `http_server/` and is launched by `scripts/start_http_server.sh`.

## Dependencies and launch

Add FastAPI and Uvicorn as project dependencies. The shell entrypoint creates
one run directory per invocation:

```text
logs/http_server/YYYYMMDD_HHMMSS/
```

The timestamp has one-second precision. The entrypoint passes the absolute run
directory to Uvicorn through an environment variable and starts the app on
`0.0.0.0:12123`. The host and port may be overridden by environment variables
for deployment and testing.

Each run directory contains:

```text
http.log
remote_policy_server.log
ssh_tunnel.log
async_policy_client.log
```

Uvicorn access logs and application errors go to `http.log`. Each child
process redirects both stdout and stderr to its own append-only log.

## API

All control routes accept `POST` only and return JSON.

### `POST /policy-server/start`

Start this command from the repository root in a new process group:

```text
scripts/start_policy_server_pi05_remote.sh configs/fold_cloth.json
```

Return the PID and log path after a successful spawn. The Policy Server is
locked from the beginning of the spawn until the process exits or is stopped.
A concurrent or later start request while it is locked returns HTTP 409 with
`Policy Server 服务正在运行`.

### `POST /policy-server/stop`

Send SIGINT to the Policy Server process group and wait for it to exit. This is
equivalent to pressing Ctrl+C in its controlling terminal and also reaches the
SSH process launched by the shell script. If it is not running, return a
successful idempotent response. If it does not exit before the configured
timeout, return an error without escalating to SIGKILL.

### `POST /policy-client/start`

Atomically lock the Policy Client workflow and start this command in a new
process group:

```text
scripts/ssh_tunnel_policy_server.sh configs/fold_cloth.json
```

Schedule the following command five seconds later in another process group:

```text
SKIP_CONFIRM=true scripts/run_async_policy_client_pi05.sh configs/fold_cloth.json
```

The route returns as soon as the tunnel is spawned. The lock covers the delay,
the tunnel, and the client. Another start request during any of those states
returns HTTP 409 with `Policy Client 服务正在运行`.

If the SSH tunnel exits during the delay, do not start the client and release
the lock. If the client later exits but the tunnel remains alive, keep the
workflow locked because it still owns a managed process.

### `POST /policy-client/stop`

First cancel any pending delayed client start. Send SIGINT to the async client
process group and wait for it to exit, then send SIGINT to the SSH tunnel
process group and wait for it to exit. This ordering prevents the client from
losing its transport before it can shut down. If nothing is running, return a
successful idempotent response.

## Process and concurrency model

A process manager owns the `subprocess.Popen` handles and protects all state
transitions with thread locks. Start checks and state changes occur under the
same lock so simultaneous FastAPI requests cannot spawn duplicates.

Children use `start_new_session=True`; shutdown targets their process groups
with `os.killpg(..., signal.SIGINT)`. Background watcher threads observe child
exit and release locks only when no process or delayed start remains for that
workflow.

Stopping during the five-second delay sets a cancellation event before doing
anything else. The delayed worker checks that event and the tunnel state before
spawning the client, preventing a stop/start race.

When the HTTP Server receives a normal shutdown signal, it stops the Policy
Client workflow first and then the remote Policy Server. It does not discover
or take ownership of processes created by earlier HTTP Server instances.

## Error handling

Spawn failures release the corresponding lock, record the exception in
`http.log`, and return HTTP 500. Unknown routes use FastAPI's normal 404 JSON
response. Invalid methods use its normal 405 response.

Stop timeouts are logged and returned as HTTP 500. Processes are not killed
forcefully because they may control robot hardware or remote SSH sessions.

## Testing

Unit tests replace process spawning, waiting, signaling, and the five-second
delay with deterministic fakes. They cover:

- exact commands, working directory, log destinations, and
  `SKIP_CONFIRM=true`;
- successful start responses and duplicate-start HTTP 409 responses;
- client start only after the delay and only while the tunnel is alive;
- cancellation during the delay;
- shutdown order: client before SSH tunnel;
- SIGINT delivery to process groups and automatic lock release;
- creation and naming of the per-launch log directory by the shell entrypoint.

Tests never open SSH connections or interact with robot hardware.
