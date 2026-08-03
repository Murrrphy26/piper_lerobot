# Piper General Operation Manual

Recording, training, and live inference share one config file, passed as the first script argument.

```bash
cd ~/code/yjw/piper
conda activate piper
```

Example configs (replace per task):

- `configs/record_pick_cube_act.json`
- `configs/fyx.json`
- `configs/record_towel_fold_act.json`

---

## 0. CAN Bring-up

Run before on-robot recording or inference:

```bash
bash scripts/bringup_can.sh
# bash scripts/reset_arms.sh   # exits teach mode; does not home pose
```

---

## 1. Recording

1. Connect arm aviation plugs and set up the task environment.
2. Edit the config: task text, `repo_id`, dataset storage path, etc.
3. Operate smoothly; for grasps, prefer stepwise pauses (arrive → pause → open → pause → lower → pause → close).

```bash
bash scripts/start_recording.sh configs/record_pick_cube_act.json
```

Wait until the terminal prints `Press Ctrl C` a second time, then start operating.

After each episode, enter a label:

- `s`: success — used for training
- `f`: failure — discarded

Do not manually delete locally recorded datasets.

---

## 2. Training

Confirm task name, dataset paths, training steps, and other hyperparameters in the config.

```bash
bash scripts/start_training.sh configs/record_pick_cube_act.json
```

Common options:

```bash
bash scripts/start_training.sh configs/record_pick_cube_act.json --dry-run
bash scripts/start_training.sh configs/record_pick_cube_act.json --no-action-compose
```

Use `tmux` if the network is unstable and you need the training process to stay alive.

---

## 3. Inference

Same config. The script moves to the dataset episode-0 start pose, then runs policy live.

```bash
bash scripts/bringup_can.sh
bash scripts/run_policy_live.sh configs/record_pick_cube_act.json
```

Skip move-to-start and infer immediately:

```bash
SKIP_MOVE_TO_START=true bash scripts/run_policy_live.sh configs/record_pick_cube_act.json
```

---

## Command Cheat Sheet

| Stage | Command |
|-------|---------|
| CAN | `bash scripts/bringup_can.sh` |
| Record | `bash scripts/start_recording.sh <config>` |
| Train | `bash scripts/start_training.sh <config>` |
| Infer | `bash scripts/run_policy_live.sh <config>` |
