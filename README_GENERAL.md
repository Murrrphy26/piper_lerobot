# Piper 通用操作手册

录制、训练、推理共用同一份 config，通过脚本第一个参数传入。

```bash
cd ~/code/yjw/piper
conda activate piper
```

示例 config（按任务替换）：

- `configs/record_pick_cube_act.json`
- `configs/fyx.json`
- `configs/record_towel_fold_act.json`

---

## 0. CAN 初始化

真机录制 / 推理前执行：

```bash
bash scripts/bringup_can.sh
# bash scripts/reset_arms.sh   # 仅推理前执行 退出示教模式，不是复原位姿
```

---

## 1. 录制

1. 接好机械臂航空接头，搭建任务环境。
2. 编辑 config：任务描述、`repo_id`、数据存储路径等。
3. 操作平稳；抓取类动作建议分步停顿（到位 → 停顿 → 开爪 → 停顿 → 落位 → 停顿 → 合爪）。

```bash
bash scripts/start_recording.sh configs/record_pick_cube_act.json
```

等待终端第二次打印 `Press Ctrl C` 后开始操作。

单次采集结束后输入标签：

- `s`：成功，参与训练
- `f`：失败，丢弃

禁止手动删除本地已录制数据集。

---

## 2. 训练

先确认 config 中任务名、数据集路径、训练步数等超参已改好。

```bash
bash scripts/start_training.sh configs/record_pick_cube_act.json
```

常用可选参数：

```bash
bash scripts/start_training.sh configs/record_pick_cube_act.json --dry-run
bash scripts/start_training.sh configs/record_pick_cube_act.json --no-action-compose
```

网络不稳定时建议用 `tmux` 保持训练进程。

---

## 3. 推理

同一 config；脚本会先回到数据集 episode0 初始位姿，再启动 live 推理。

```bash
bash scripts/bringup_can.sh
bash scripts/run_policy_live.sh configs/record_pick_cube_act.json
```

跳过归位、直接推理：

```bash
SKIP_MOVE_TO_START=true bash scripts/run_policy_live.sh configs/record_pick_cube_act.json
```

---

## 命令速查

| 阶段 | 命令 |
|------|------|
| CAN | `bash scripts/bringup_can.sh` |
| 录制 | `bash scripts/start_recording.sh <config>` |
| 训练 | `bash scripts/start_training.sh <config>` |
| 推理 | `bash scripts/run_policy_live.sh <config>` |
