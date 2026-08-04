# 数据预处理方案说明

本文档总结 Piper 项目中针对「夹取橙色方块」任务引入的数据预处理与增强方案，包括代码改动、使用流程，以及后续参数调整建议。

---

## 1. 方案目标

| 预处理项 | 目的 | 适用数据 |
|----------|------|----------|
| **视频色彩增强** | 校正白平衡、提升对比度、强化橙色方块可见性 | 三路相机 RGB 图像 |
| **机械臂平滑（防抖）** | 降低关节/夹爪读数与示教 action 的高频抖动 | `observation.state`（14 维）与 `action`（14 维） |

设计原则：**训练数据与推理输入必须使用同一套预处理配置**，否则策略看到的分布不一致，效果会明显变差。

---

## 2. 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│ 录制 / 离线 / 推理 共用 preprocessing.py + JSON 配置          │
└─────────────────────────────────────────────────────────────┘

录制 (record_episode)          离线 (preprocess_dataset)       推理 (run_policy_live)
        │                                │                              │
        ▼                                ▼                              ▼
  EMA 因果平滑                     Savitzky-Golay 整段平滑          EMA 因果平滑
  + 图像增强                       + 图像增强                       + 图像增强
        │                                │                              │
        └──────────────┬─────────────────┴──────────────────────────────┘
                       ▼
              LeRobot 数据集 (parquet + video)
                       ▼
              lerobot-train (ACT)   ← 训练阶段不做额外 transform
                       ▼
              checkpoint → policy_live（需开启相同 preprocessing）
```

**注意：** `start_training.py` 只负责校验数据集、筛选 episode、调用 `lerobot-train`，**不会在训练时动态应用预处理**。预处理结果必须已经写入数据集，或在录制/推理时在线应用。

---

## 3. 代码改动清单

### 3.1 新增文件

| 文件 | 说明 |
|------|------|
| `src/piper_train/preprocessing.py` | 核心模块：图像增强、EMA/SG 平滑、`FramePreprocessor` |
| `tools/preprocess_dataset.py` | 离线批处理：读取原 LeRobot 数据集，写出新 `repo_id` |
| `docs/data-preprocessing.md` | 本文档 |

### 3.2 修改文件

| 文件 | 改动 |
|------|------|
| `configs/record_pick_cube.json` | 新增 `preprocessing` 配置段 |
| `src/piper_train/record_episode.py` | 在 `record_frame()` 前调用 `FramePreprocessor` |
| `src/piper_train/start_recording.py` | 从 JSON 读取 `preprocessing`，跳过 `training`/`policy_live` 等非录制字段 |
| `src/piper_train/run_policy_live.py` | 策略输入前对 observation 应用相同预处理 |
| `src/piper_train/start_policy_live.py` | 从顶层或 `policy_live` 读取 `preprocessing` |

### 3.3 未改动的部分

- `start_training.py`：训练流程不变，仍调用外部 `lerobot-train`
- `recorder.py`：数据 schema 不变（14 维 state/action + 多路图像）
- `episode_outcomes.py`：episode 成功/失败筛选逻辑不变

---

## 4. 预处理算法说明

### 4.1 图像流水线（按顺序）

1. **灰度世界白平衡（Gray World）**  
   估计 RGB 三通道均值，缩放使整体趋近中性灰，补偿偏色。

2. **CLAHE（LAB 空间 L 通道）**  
   自适应直方图均衡，提升局部对比度，便于阴影/高光下识别方块。

3. **HSV 橙色增强**  
   在 `hue_range` 内像素提高饱和度与明度，突出橙色目标。

依赖：`opencv-python`（必须）。

### 4.2 机械臂平滑

| 场景 | 方法 | 说明 |
|------|------|------|
| 录制 / 真机推理 | **EMA（指数滑动平均）** | 因果滤波，不使用未来帧 |
| 离线数据集处理 | **Savitzky-Golay** | 整段 episode 平滑，保形更好；无 `scipy` 时退化为滑动平均 |

平滑对象：

- `observation.state`：从臂 6+1 + 6+1 关节/夹爪
- `action`：主臂示教动作（与 state 使用相同滤波参数）

夹爪通道（索引 6、13）可单独设置更大的 `gripper_ema_alpha`（更跟手 vs 更平滑）。

### 4.3 与推理层 `smooth_action` 的关系

| 层级 | 位置 | 作用 |
|------|------|------|
| **数据预处理** | 录制/离线/推理输入 | 平滑 observation（及录制时的 action） |
| **策略输出平滑** | `policy_live.smoothing_alpha` | 平滑模型预测的 action，防止真机突变 |

两层互补，不互相替代。

---

## 5. 配置参考

配置写在 `configs/record_pick_cube.json` 的 `preprocessing` 段，录制与推理共用：

```json
"preprocessing": {
  "enabled": true,
  "dual_record": {
    "enabled": true,
    "augmented_repo_id": "local/cube_v4_3view_side_pp"
  },
  "images": {
    "enabled": true,
    "white_balance": true,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": 8,
    "orange_boost": {
      "enabled": true,
      "hue_range": [5, 25],
      "sat_scale": 1.2,
      "val_scale": 1.05
    }
  },
  "smoothing": {
    "enabled": true,
    "method": "ema",
    "ema_alpha": 0.25,
    "gripper_ema_alpha": 0.35,
    "savgol_window": 7,
    "savgol_polyorder": 2
  }
}
```

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `enabled` | bool | `false` | 总开关；`false` 时全部跳过 |
| `images.enabled` | bool | `true` | 图像子开关 |
| `images.white_balance` | bool | `true` | 是否做白平衡 |
| `images.clahe_clip_limit` | float | `2.0` | CLAHE 对比度限制；`0` 可关闭 CLAHE |
| `images.clahe_tile_grid_size` | int | `8` | CLAHE 分块大小 |
| `orange_boost.enabled` | bool | `true` | 是否做橙色 HSV 增强 |
| `orange_boost.hue_range` | [int, int] | `[5, 25]` | OpenCV HSV 色相范围（0–179） |
| `orange_boost.sat_scale` | float | `1.2` | 橙色区域饱和度倍率 |
| `orange_boost.val_scale` | float | `1.05` | 橙色区域明度倍率 |
| `smoothing.enabled` | bool | `true` | 关节平滑子开关 |
| `smoothing.method` | str | `"ema"` | 在线用 `ema`；离线脚本可覆盖为 `savgol` |
| `smoothing.ema_alpha` | float | `0.25` | EMA 系数，越大越跟手、越小越平滑 |
| `smoothing.gripper_ema_alpha` | float | `0.35` | 夹爪单独 EMA；`null` 则与关节相同 |
| `smoothing.savgol_window` | int | `7` | SG 窗口长度（奇数，约 0.5–0.9s @10fps） |
| `smoothing.savgol_polyorder` | int | `2` | SG 多项式阶数 |
| `dual_record` | bool 或 object | `true` | 开启时同时写入原始与增强两个数据集 |
| `dual_record.enabled` | bool | `true` | 双目录录制开关（需 `preprocessing.enabled: true`） |
| `dual_record.augmented_repo_id` | str | `{repo_id}_pp` | 增强版数据集 repo id |

---

## 5.1 双目录录制（推荐）

当 `preprocessing.enabled: true` 且 `dual_record.enabled: true`（默认开启）时，每次录制会**并行写入两个 LeRobot 数据集**：

| 数据集 | 配置项 | 内容 |
|--------|--------|------|
| **原始** | 顶层 `repo_id` | 相机原图 + 原始关节/action |
| **增强** | `dual_record.augmented_repo_id` | 色彩增强 + 平滑后的数据 |

目录示例：

```
data/lerobot/local/cube_v4_3view_side/       ← 原始 video
data/lerobot/local/cube_v4_3view_side_pp/    ← 增强 video
```

episode 标注（`episode_outcomes.jsonl`）会**同时写入两个数据集**，episode 索引保持对齐。

关闭双目录录制（回退到只写一个数据集）：

```json
"dual_record": false
```

或只写增强版到 `repo_id`（旧行为）：

```json
"dual_record": { "enabled": false }
```


## 6. 使用流程

### 6.1 是否需要重新录制？

**不一定。** 三种路径：

| 路径 | 操作 | 适用 |
|------|------|------|
| **A. 离线预处理（推荐）** | 对现有数据集跑 `preprocess_dataset.py` | 已有大量 episode，不想重录 |
| **B. 原始数据 baseline** | 关闭 `preprocessing.enabled`，直接训练 | 对比实验、快速验证 |
| **C. 重新录制** | 开启 `preprocessing.enabled` 后新录 | 希望从源头统一预处理 |

### 6.2 路径 A：离线预处理 + 训练 + 推理

```bash
# 1. 生成预处理数据集（不覆盖原数据）
python tools/preprocess_dataset.py \
  --config configs/record_pick_cube.json \
  --target-repo-id local/cube_v4_3view_side_pp \
  --smoothing-method savgol

# 2. 修改 configs/record_pick_cube.json：
#    repo_id → local/cube_v4_3view_side_pp
#    training.output_dir / job_name → 带 _pp 后缀
#    policy_live.repo_id / dataset_root / policy_path → 对应更新
#    preprocessing.enabled → true

# 3. 训练
bash scripts/start_training.sh configs/record_pick_cube.json

# 4. 真机推理
bash scripts/run_policy_live.sh configs/record_pick_cube.json
```

### 6.3 路径 B：原始数据训练

```json
"preprocessing": { "enabled": false }
```

直接 `bash scripts/start_training.sh`；推理时也保持 `enabled: false`。

### 6.4 路径 C：录制时已预处理

```bash
bash scripts/start_recording.sh configs/record_pick_cube.json
```

确保 `preprocessing.enabled: true`，写入的数据集已是处理后版本。

### 6.5 完整训练流程

```
数据集 (LeRobot)
    ↓
start_training.py
    ├─ validate_dataset()        检查 meta/info.json、parquet
    ├─ patch_image_feature_names()
    ├─ episode_outcomes 筛选     默认 exclude-failures
    └─ subprocess: lerobot-train
    ↓
outputs/train/{job_name}/checkpoints/{step}/pretrained_model
    ↓
start_policy_live / run_policy_live
    └─ 与训练数据一致的 preprocessing + policy_live.smoothing_alpha
```

### 6.6 训练常见问题

**输出目录已存在：**

```
FileExistsError: Output directory outputs/train/... already exists and resume is False
```

处理方式（三选一）：

1. 续训：在 `lerobot-train` 加 `--resume=true`
2. 删除旧目录：`rm -rf outputs/train/act_cube_v4_3view_side`
3. 改配置：`training.output_dir` / `job_name` 换成新名字

---

## 7. 参数调整指导

### 7.1 图像：橙色方块不明显

| 现象 | 建议调整 |
|------|----------|
| 方块偏红/偏黄，增强区域不准 | 收窄或平移 `hue_range`，如 `[8, 20]` |
| 整体偏暗 | 略增 `val_scale`（1.05 → 1.10），或 `clahe_clip_limit`（2.0 → 2.5） |
| 噪声/过曝 | 降低 `clahe_clip_limit`（→ 1.5）或 `sat_scale`（→ 1.1） |
| 白平衡过度偏色 | 设 `white_balance: false`，仅保留 CLAHE + 橙色增强 |

**调试方法：** 从数据集中抽几帧，对比原图与 `enhance_image()` 输出；或先用 `--dry-run` 看离线脚本计划，再处理 1 个 episode 目视检查。

### 7.2 图像：多相机一致性

三路相机（`cam_right` / `cam_side` / `cam_top`）共用同一套参数。若顶视与侧视光照差异大：

- 优先调 `clahe_clip_limit` 和 `white_balance`（全局稳定）
- 橙色增强幅度不宜过大（`sat_scale` > 1.5 易失真）

### 7.3 机械臂：轨迹抖动仍明显

| 参数 | 方向 | 效果 |
|------|------|------|
| `ema_alpha` ↓ | 0.25 → 0.15 | 更平滑，滞后更大 |
| `ema_alpha` ↑ | 0.25 → 0.35 | 更跟手，抖动残留更多 |
| `gripper_ema_alpha` ↓ | 0.35 → 0.25 | 夹爪开合更稳、略慢 |
| 离线 `savgol_window` ↑ | 7 → 9 | 整段更平滑（仅离线） |

录制 10 fps 时，`savgol_window=7` 约 0.7 s 窗口；窗口过大可能抹平快速抓取动作。

### 7.4 机械臂：动作发「钝」、跟不上

- 增大 `ema_alpha`（0.25 → 0.30）
- 离线训练改用较小 `savgol_window`（7 → 5）或 `savgol_polyorder` 保持 2
- 检查是否同时开了过大的 `policy_live.smoothing_alpha`（默认 0.25）

### 7.5 训练 / 推理一致性检查清单

- [ ] 训练用的 `repo_id` 是否为预处理后的数据集（若 `preprocessing.enabled: true`）
- [ ] 推理时 `preprocessing` 与训练数据生成方式一致
- [ ] 若训练用原始数据，推理必须 `preprocessing.enabled: false`
- [ ] `policy_live.dataset_root` 指向训练时同一数据集（供 stats / preprocessor）
- [ ] 修改预处理后需**重新训练**，旧 checkpoint 不适用新分布

### 7.6 对比实验建议

1. **Baseline**：原始数据 + `preprocessing.enabled: false`
2. **仅图像**：离线处理时可在代码中临时关 smoothing，或设 `smoothing.enabled: false`
3. **仅平滑**：设 `images.enabled: false`，保留 smoothing
4. **完整方案**：图像 + 平滑（当前默认）

每次只改一组参数，固定 `training.output_dir`，便于对比 success rate 与离线 `offline_infer` 误差。

---

## 8. 数据格式（未变）

LeRobot 每帧仍为：

- `observation.state`：`float32[14]`
- `action`：`float32[14]`
- `observation.images.{cam_name}`：RGB `uint8`，默认 640×360
- `task`：字符串

预处理**不改变 schema**，只改变数值/像素内容。

---

## 9. 依赖

| 包 | 用途 | 必须 |
|----|------|------|
| `opencv-python` | 白平衡、CLAHE、HSV | 是 |
| `scipy` | 离线 Savitzky-Golay | 否（无则滑动平均替代） |
| `numpy` | 数组运算 | 是 |

---

## 10. 相关命令速查

```bash
# 录制（在线预处理）
bash scripts/start_recording.sh configs/record_pick_cube.json

# 离线预处理
python tools/preprocess_dataset.py --config configs/record_pick_cube.json \
  --target-repo-id local/cube_v4_3view_side_pp --smoothing-method savgol

# 仅查看训练命令
python -m piper_train.start_training --config configs/record_pick_cube.json --dry-run

# 训练
bash scripts/start_training.sh configs/record_pick_cube.json

# 真机推理
bash scripts/run_policy_live.sh configs/record_pick_cube.json

# 离线验证 checkpoint
python -m piper_train.offline_infer \
  --policy-path outputs/train/act_cube_v4_3view_side/checkpoints/010000/pretrained_model \
  --dataset-root data/lerobot/local/cube_v4_3view_side_pp
```

---

## 11. 后续可扩展方向（未实现）

- 在 `start_training.py` 中透传 `resume` 配置项
- 训练时随机增强（需 LeRobot dataset transform 或自定义 Dataset）
- 按相机单独配置色彩参数
- 预处理前后自动生成对比图/report 工具

---

*文档版本：与 `preprocessing.py` 及 `configs/record_pick_cube.json` 初始实现同步。*
