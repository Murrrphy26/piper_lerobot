# Piper 6-DoF Arm PI05 Operation Manual

# Quick Start

## 1. Initialize the Industrial PC CAN Bus

```bash
cd ~/code/yjw/piper
conda activate piper
bash scripts/bringup_can.sh
```

## 2. Record Datasets Locally

1. Hardware prep: Connect the aviation connectors for all four arms and set up the task environment.

2. Config: Open the config file and update the task name and data storage path for this session.

3. Collection requirements:

- Place blocks as randomly as possible during recording to improve data generalization.

- Operate smoothly and under control throughout. Perform every grasp as **stepwise pauses** — do not run continuous, fast motions. Standard sequence: move above the block → pause and settle → open gripper → pause and settle → lower into place → pause and settle → close gripper → pause and settle. Finish one grasp before continuing.

Start the recording script (default config: orange block pick task):

```bash
bash scripts/start_recording_pi05.sh
```

Wait until the terminal prints `Press Ctrl C` a second time, then begin operating.

After each episode, enter a data label:

- `s`: task succeeded — data is used for later training

- `f`: task failed — data is discarded and not used for training

Do not manually delete locally recorded datasets.

After all recording is done, package and upload the dataset to the remote server:

```bash
bash scripts/upload_dataset_to_remote.sh
```

## 3. Model Training on the Remote Server

Network can be unstable; use a tmux session to keep the training process alive:

```bash
ssh allinai2
cd /mnt/disk/fyx/piper
conda activate piper
```

Project root: `/mnt/disk/fyx/piper`

Edit the config: task name, log path, training steps, and other hyperparameters.

Start training:

```bash
bash scripts/start_training_pi05.sh
```

## 4. Remote Async Inference (strict order — do not skip steps)

### Step 1: Start the Policy Server on the remote GPU host (keep this terminal running)

```bash
ssh allinai2
cd /mnt/disk/fyx/piper
bash scripts/start_policy_server_pi05.sh
```

**Wait for the full log output below before continuing**:

```bash
INFO ... Preloading policy at startup | type=pi05 | path=outputs/train/...
INFO ... Startup preload finished in XXXs
INFO ... PolicyServer started on 0.0.0.0:8080
```

Note: With startup preload enabled (`policy_server.preload_at_startup: true`), the first launch takes about 1–3 minutes to load the model. The service listens on port 8080 only after loading finishes.

### Step 2: On the industrial PC, open a new terminal and create an SSH tunnel (do not close this terminal)

```bash
cd ~/code/yjw/piper
conda activate piper
bash scripts/ssh_tunnel_policy_server.sh
```

Example of a successful output:

```bash
Opening SSH tunnel for policy server gRPC.
  local:  127.0.0.1:8080
  remote: 127.0.0.1:8080
  ssh host: allinai2
```

Closing this terminal will break the inference communication link.

### Step 3: On the industrial PC, open a new terminal and start the robot inference client

```bash
cd ~/code/yjw/piper
conda activate piper
```

Before inference, manually initialize the CAN bus and arms (the client script does not bring up hardware automatically):

```bash
bash scripts/bringup_can.sh
# bash scripts/reset_arms.sh   # exits teach mode; does not home/reset pose
```

Start the remote async inference client:

```bash
bash scripts/run_async_policy_client_pi05_remote.sh
```

# Troubleshooting

## 1. Data Recording

1. Running `scripts/start_policy_server_pi05.sh` raises OpenCV-related errors

Cause: camera stream latency conflict. Fix: run the script once more.

## 2. Async Inference

1. High network latency causes stuttering inference and lagged arm motion

Diagnostic command (run on the industrial PC):

```bash
ssh allinai2 "ping 127.0.0.1 -c 10"
```

Use ping latency to judge link quality between the server and the industrial PC. If latency is high, switch to a wired network or restart the SSH tunnel.
