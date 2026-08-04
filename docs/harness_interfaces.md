# Piper Harness interfaces

本文档记录当前 harness 相关层次的关键接口。目标是让夹方块、毛巾、叠衣服等任务共享同一套运行底座，同时允许叠衣服任务逐步启用视觉标注、阶段管理和更强的恢复逻辑。

当前设计原则：

- 夹方块等简单任务默认不启用 harness，保持原始视觉直通和单 policy 推理。
- 叠衣服任务启用 harness 时，必须启用视觉标注。
- VLA/ACT 仍是主要动作执行者；harness 先做观测标注、阶段判断、安全过滤和日志，而不是抢完整控制权。

## 1. 视觉输入 / 视觉标注层

### 职责

视觉层分两条路径：

```text
camera RGB
  ├── 原图直通 policy
  └── 可选视觉标注 → vision_state
```

简单任务只使用原图直通。叠衣服 harness 使用视觉标注结果做阶段判断、日志和后续纠错分析。

### 当前实现

```text
src/piper_train/cloth_vision.py
src/piper_train/annotate_cloth_vision.py
```

`cloth_vision.py` 当前实现的是白色衣服 / 黄色木纹桌面的轻量视觉标注：

```text
RGB
  → HSV 低饱和高亮阈值
  → RGB 均值/通道差约束
  → 形态学 open/close
  → 最大连通域
  → bbox / center / PCA orientation / compactness / extent
```

### 主要接口

```python
@dataclass(frozen=True)
class WhiteClothVisionConfig:
    max_saturation: int = 80
    min_value: int = 135
    min_rgb_mean: int = 130
    max_rgb_channel_spread: int = 70
    blur_kernel: int = 5
    morph_kernel: int = 7
    morph_iterations: int = 2
    min_area_ratio: float = 0.005
    roi_norm: tuple[float, float, float, float] | None = None
```

```python
@dataclass
class ClothVisionState:
    valid: bool
    image_width: int
    image_height: int
    area_px: int
    area_ratio: float
    bbox_xywh: list[int] | None
    center_xy: list[float] | None
    orientation_deg: float | None
    compactness: float | None
    extent: float | None
```

```python
@dataclass(frozen=True)
class ClothGeometryThresholds:
    min_area_ratio_for_visible: float = 0.01
    min_area_ratio_for_flat: float = 0.12
    min_extent_for_flat: float = 0.35
    min_compactness_for_flat: float = 0.55
    target_center_norm: tuple[float, float] = (0.5, 0.5)
    max_center_error_norm: float = 0.18
    target_orientation_deg: float = 0.0
    max_orientation_error_deg: float = 20.0
```

```python
@dataclass
class ClothGeometryChecks:
    visible: bool
    flat_enough: bool
    centered_enough: bool
    oriented_enough: bool
    aligned_enough: bool
    center_error_norm: float | None
    orientation_error_deg: float | None
```

```python
class ClothVisionAnnotator:
    def annotate(self, rgb: np.ndarray) -> ClothVisionResult:
        ...
```

`ClothVisionResult` 包含：

- `state`：原始几何状态；
- `checks`：可直接给 stage checker 用的布尔判断；
- `mask`：用于 debug overlay，不建议写入每步在线日志。

```python
def estimate_cloth_state(
    rgb: np.ndarray,
    config: WhiteClothVisionConfig | None = None,
) -> tuple[ClothVisionState, np.ndarray]:
    ...
```

返回：

- `ClothVisionState`：结构化视觉状态；
- `mask`：最大连通衣服区域，`uint8`，0/255。

### 调试入口

```bash
python -m piper_train.annotate_cloth_vision --camera 16
python -m piper_train.annotate_cloth_vision --image camera_temp/live.jpg
```

实时调参：

```bash
python -m piper_train.annotate_cloth_vision --camera 16 --watch
```

输出：

```text
outputs/vision_debug/*_state.json
outputs/vision_debug/*_mask.png
outputs/vision_debug/*_overlay.jpg
```

### 后续接入 harness 的方式

未来 runner 每帧读取 observation 后：

```python
rgb = observation["cam_top"]
vision_state, mask = estimate_cloth_state(rgb, cloth_vision_config)
```

`vision_state` 不直接替代 policy 输入；policy 仍吃原图。`vision_state` 给阶段管理、成功检查和日志使用。

## 2. 技能策略层

### 职责

策略层负责：

```text
observation.state + observation.images + task/prompt
  → policy
  → predicted_action 或 action chunk
```

当前支持：

- ACT / `act_piper`
- pi05 async policy server/client

### 当前实现

同步 live：

```text
src/piper_train/run_policy_live.py
src/piper_train/start_policy_live.py
```

异步 pi05：

```text
src/piper_train/async_robot_client.py
src/piper_train/async_policy_server.py
src/piper_train/start_async_policy_client.py
src/piper_train/start_async_policy_server.py
```

自定义 ACT policy 插件：

```text
src/lerobot_policy_act_piper/
```

其中 `act_piper` 是 LeRobot ACT 的 Piper 多相机增强版本，支持：

- per-camera scales；
- camera id embedding；
- 原 ACT pre/post processor。

### 主要接口

同步 policy input：

```python
def observation_to_policy_input(
    observation: dict[str, Any],
    camera_names: list[str],
    task: str,
) -> dict[str, Any]:
    ...
```

输出 action dict：

```python
{
  "left_joint_1.pos": float,
  ...
  "left_gripper.pos": float,
  "right_joint_1.pos": float,
  ...
  "right_gripper.pos": float,
}
```

ACT action chunk 入口：

```python
policy.predict_action_chunk(processed)
```

pi05 async action 由 remote policy server 返回 action queue，client 端逐步 pop。

### 后续 harness 接法

阶段管理层可以只改变：

- active policy；
- prompt/task；
- control overrides；
- timeout / checker。

不需要改变底层 observation/action 格式。

## 3. 动作执行与安全层

### 职责

动作层位于 policy 输出和硬件发送之间：

```text
predicted_action
  → smoothing
  → safety filter
  → robot.send_action()
```

它负责：

- EMA 平滑；
- 可选 FK 工作空间检查；
- 可选 NaN/inf 检查；
- 记录 action 被修改/阻挡的原因；
- 统一同步 ACT 和异步 pi05 的动作后处理。

### 当前实现

```text
src/piper_train/action_pipeline.py
src/piper_train/piper.py
```

`piper.py` 仍负责底层硬件发送：

- SDK `MotionCtrl_2` 速度档；
- 逐关节 `max_joint_step_rad`；
- 夹爪 `max_gripper_step_m`；
- `JointCtrl` / `GripperCtrl`。

`action_pipeline.py` 负责新增的公共 action post-processing。

### 主要接口

```python
def smooth_action(
    action: dict[str, float],
    previous_action: dict[str, float] | None,
    alpha: float,
) -> dict[str, float]:
    ...
```

```python
@dataclass(frozen=True)
class WorkspaceSideLimit:
    min_z_m: float | None = None
    allowed_below_min_m: float = 0.0
```

`allowed_below_min_m` 表示可允许末端超过标定最低点多少距离：

```text
effective_min_z = min_z_m - allowed_below_min_m
```

```python
@dataclass(frozen=True)
class ActionSafetyConfig:
    enabled: bool = False
    on_violation: str = "hold_previous"  # hold_previous | warn | stop
    left: WorkspaceSideLimit
    right: WorkspaceSideLimit
    finite_check: bool = True
```

```python
class ActionPipeline:
    def process(
        self,
        predicted_action: dict[str, float],
        previous_action: dict[str, float] | None,
        alpha: float,
    ) -> ActionPipelineResult:
        ...
```

返回：

```python
@dataclass
class ActionPipelineResult:
    smoothed_action: dict[str, float]
    filtered_action: dict[str, float]
    events: list[dict[str, Any]]
    blocked: bool
```

### FK provider

```python
class FKProvider(Protocol):
    def ee_pose(self, side: str, joints_rad: list[float]) -> EEPose | None:
        ...
```

当前实现：

```python
class PiperSDKFKProvider:
    ...
```

使用：

```text
piper_sdk.C_PiperForwardKinematics.CalFK(joints_rad)
```

SDK 约定：

- 输入关节角：radian；
- 输出 xyz：mm；
- 输出 rpy：degree；
- 当前 provider 取最后一节 link 的 xyz，并转换为 m。

### workspace 标定入口

```text
src/piper_train/calibrate_workspace.py
```

示例：

```bash
python -m piper_train.calibrate_workspace configs/record_towel_fold_act.json \
  --side right \
  --output configs/calibration/workspace_towel_right.json \
  --allowed-below-min-m 0.005
```

输出文件包含可复制/引用的 safety block：

```json
{
  "safety": {
    "enabled": true,
    "fk_provider": "piper_sdk",
    "dh_is_offset": 1,
    "allowed_below_min_m": 0.005,
    "on_violation": "hold_previous",
    "right_min_z_m": 0.034
  }
}
```

## 4. 日志层

### 职责

日志层用于区分：

```text
模型想做什么
平滑后变成什么
安全层过滤后变成什么
最终硬件收到什么
```

这对判断“模型没学会”还是“执行层抹掉动作”非常重要。

### 当前实现

```text
src/piper_train/run_policy_live.py
src/piper_train/async_robot_client.py
```

同步 live 还有：

```text
LiveRunRecorder
```

用于汇总跟踪误差、chunk 情况、loop timing 等。

### 当前关键字段

每步日志：

```json
{
  "current_action": {},
  "predicted_action": {},
  "smoothed_action": {},
  "filtered_action": {},
  "sent_action": {},
  "safety_events": [],
  "safety_blocked": false
}
```

后续 harness 应新增：

```json
{
  "stage": "unfold",
  "prompt": "unfold the shirt",
  "vision_state": {},
  "checker_result": {},
  "event": "stage_start | stage_success | stage_failure | human_intervention_start"
}
```

## 5. 阶段管理层（计划中）

### 职责

阶段管理层只在 harness 启用时运行。夹方块等简单任务不需要该层。

目标：

```text
vision_state + time + policy status
  → current stage
  → prompt / policy / control params
  → transition or retry
```

### 预期 stage schema

示例：

```json
{
  "name": "unfold",
  "policy_ref": "main",
  "prompt": "unfold the shirt until it lies flat",
  "max_duration_s": 20,
  "success_checker": "cloth_flat_enough",
  "next_on_success": "align",
  "next_on_failure": "recovery",
  "control_overrides": {
    "control_speed": 50,
    "smoothing_alpha": 0.6,
    "control_hz": 20
  }
}
```

### 最小阶段设计

叠衣服第一版：

```text
UNFOLD
ALIGN
FOLD
DONE
```

推荐先支持手动/半自动 transition：

- 手动按键切阶段；
- 视觉状态只记录；
- 之后再启用自动 checker。

## 6. 当前边界

已经实现：

- `piper_train` 包名；
- action pipeline；
- SDK FK provider；
- workspace 标定；
- 同步/异步安全日志；
- 白衣服视觉标注 v1；
- 视觉调试 CLI。

尚未实现：

- harness runner；
- stage manager；
- 多阶段 prompt/control override；
- VLM/语义关键点；
- 自动 DAgger intervention 标记；
- 视觉状态接入 live 日志。
