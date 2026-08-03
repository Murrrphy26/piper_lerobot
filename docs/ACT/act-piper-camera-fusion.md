# ACT Piper 多相机融合改动方案

本文档说明针对「夹取橙色方块」任务，在 LeRobot ACT 基础上引入**静态相机特征缩放**与**相机身份 Embedding（Transformer 注意力改造 3A）**的完整改动方案。

---

## 1. 背景与目标

### 1.1 问题

方块抓取任务使用三路相机：

| 相机 key | 角色 | 说明 |
|----------|------|------|
| `cam_right` | **腕相**（精细对准主视角） | 夹爪附近，方块边缘细节最清晰 |
| `cam_side` | 侧视 | 辅助判断高度 |
| `cam_top` | 俯视 | 全局粗定位 |

原版 LeRobot ACT 对三路相机**等权处理**：共享 ResNet backbone，特征展平为 spatial token 后直接 concat 进 Transformer encoder，无相机区分。俯视大范围场景容易在 self-attention 中掩盖腕相的小范围方块特征，导致夹取偏移。

### 1.2 本次实现的两项改动

| 方案 | 原理 | 状态 |
|------|------|------|
| **静态相机缩放权重** | 投影后对每相机 feature map 乘标量 α，放大腕相 token 幅值 | ✅ 已实现 |
| **相机 ID Embedding（3A）** | 每相机 token 加可学习身份向量，让 attention 区分视图类型 | ✅ 已实现 |
| StateGate 动态门控（方案二） | 按距离/夹爪状态动态调权 | ⏳ 未实现 |
| Attention mask（方案 3B） | 限制俯视对腕相的注意力 | ⏳ 未实现 |

### 1.3 设计原则

- **不 fork LeRobot 源码**：以第三方 policy 插件（`lerobot_policy_act_piper`）形式注册，训练仍走 `lerobot-train`
- **不改数据集 schema**：完全复用现有轨迹，无需重录
- **推理零改动**：装好插件后，`offline_infer` / `run_policy_live` 按 checkpoint 的 `type: act_piper` 自动加载

---

## 2. 架构对比

### 2.1 原版 ACT（`policy.type=act`）

```
每相机图像
  → 共享 ResNet18 backbone
  → 1×1 conv 投影 (dim_model)
  → 展平为 H×W 个 spatial token
  → 与 latent / state token concat
  → Transformer Encoder (4层 self-attn)
  → Decoder (1层 cross-attn)
  → action_head
```

特点：三相机 token **无差别**进入 self-attention；2D 正弦位置编码只编码空间位置，不编码相机身份。

### 2.2 ACT Piper（`policy.type=act_piper`）

在「1×1 conv 投影之后、展平进 Transformer 之前」插入两步：

```
feat = proj(backbone(img)) × α_cam + camera_id_embed[cam_idx]
```

完整数据流：

```
┌─────────────────────────────────────────────────────────────┐
│  latent token + robot state token                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    ▼                       ▼                       ▼
 cam_right              cam_side                 cam_top
 (腕相 α=2.0)           (侧视 α=0.9)            (俯视 α=0.8)
    │                       │                       │
 ResNet → proj → ×α → +embed → H×W tokens (各自独立)
    │                       │                       │
    └───────────────────────┴───────────────────────┘
                            │
                            ▼
              Transformer Encoder → Decoder → action
```

---

## 3. 改动细节

### 3.1 新增：`lerobot_policy_act_piper` 插件包

| 文件 | 职责 |
|------|------|
| `src/lerobot_policy_act_piper/configuration_act_piper.py` | 注册 `act_piper` 配置类，扩展 ACT 超参 |
| `src/lerobot_policy_act_piper/modeling_act_piper.py` | `ACTPiper` 模型 + `ACTPiperPolicy` 策略包装 |
| `src/lerobot_policy_act_piper/processor_act_piper.py` | 复用 ACT 归一化 preprocessor |
| `src/lerobot_policy_act_piper/__init__.py` | 插件入口，触发 LeRobot 注册 |
| `pyproject.toml` | 包名 `lerobot_policy_act_piper`（pip 可编辑安装） |

### 3.2 修改：Piper 训练启动

| 文件 | 改动 |
|------|------|
| `src/piper_towel_fold/start_training.py` | 新增 `append_act_piper_policy_options()`，向 `lerobot-train` 透传 `--policy.camera_scales` 等参数 |
| `configs/record_pick_cube.json` | `policy_type` 改为 `act_piper`，写入默认相机权重与输出路径 |

### 3.3 未改动部分

- `recorder.py` / 数据 schema
- `preprocessing.py` / 预处理流水线
- `offline_infer.py` / `run_policy_live.py`（推理逻辑不变）
- LeRobot 安装目录内的 `modeling_act.py`

---

## 4. 核心算法

### 4.1 静态相机缩放（Per-camera Feature Scaling）

**切入位置**：`encoder_img_feat_input_proj` 之后、`einops.rearrange` 展平之前。

```python
cam_features = self.encoder_img_feat_input_proj(self.backbone(img)["feature_map"])
cam_features = cam_features * self.camera_scales[cam_idx]  # 标量广播到 (B,C,H,W)
```

**系数解析**：按 `config.image_features` 顺序，用短相机名查表：

```python
# image_features 示例：
# ["observation.images.cam_right", "observation.images.cam_side", "observation.images.cam_top"]
scales.append(config.camera_scales.get("cam_right", 1.0))  # → 2.0
```

**存储方式**：

- `learnable_camera_scales=false`（默认）：`register_buffer`，训练全程固定
- `learnable_camera_scales=true`：`nn.Parameter`，与模型一起优化

**推荐初值（方块抓取）**：

| 相机 | α | 理由 |
|------|---|------|
| `cam_right`（腕相） | 2.0 | 精细对准核心，放大 token 幅值 |
| `cam_side` | 0.9 | 高度辅助，略压低 |
| `cam_top` | 0.8 | 俯视仅粗定位，降低全局主导 |

### 4.2 相机 ID Embedding（方案 3A）

**切入位置**：展平为 `(H×W, B, dim_model)` 之后，extend 进 token 列表之前。

```python
cam_id = self.camera_id_embed.weight[cam_idx]       # (dim_model,)
cam_features = cam_features + cam_id.view(1, 1, -1) # 广播到所有 spatial token
```

**作用**：

- 原版 ACT 的 `ACTSinusoidalPositionEmbedding2d` 只编码像素在图内的 2D 位置
- 相机 ID embedding 为每路相机提供**可学习的视图类型标识**
- Transformer self-attention 可据此学到「靠近时依赖腕相、远离时依赖俯视」等规则

**与缩放的关系**：

- α：人工先验偏置（腕相更重要）
- camera_id_embed：数据驱动的可学习身份区分
- 两者互补，默认**同时开启**

### 4.3 为何不改 Decoder / Attention Mask

- Decoder 仅 1 层（对齐原版 ACT），encoder 有 4 层 self-attn，改 encoder 输入已足够
- Attention mask（方案 3B）需改 `MultiheadAttention` 接口并维护 token 归属表，侵入性高
- 静态缩放 + ID embed 改动量小、可独立 ablation，优先验证效果

---

## 5. 配置说明

### 5.1 `configs/record_pick_cube.json` 训练段

```json
"training": {
  "policy_type": "act_piper",
  "job_name": "act_piper_cube_v4_3view_side",
  "output_dir": "outputs/train/act_piper_cube_v4_3view_side",
  "camera_scales": {
    "cam_right": 2.0,
    "cam_side": 0.9,
    "cam_top": 0.8
  },
  "learnable_camera_scales": false,
  "use_camera_id_embed": true,
  "steps": 10000,
  "batch_size": 4,
  ...
}
```

### 5.2 配置项说明

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `policy_type` | string | — | 必须为 `act_piper` |
| `camera_scales` | dict | 见上表 | key 为短相机名（非 `observation.images.*`） |
| `learnable_camera_scales` | bool | `false` | 是否将 α 设为可学习参数 |
| `use_camera_id_embed` | bool | `true` | 是否启用相机身份 embedding |

### 5.3 透传到 `lerobot-train` 的 CLI 参数

`start_training.py` 自动生成：

```bash
--policy.type=act_piper
--policy.camera_scales={"cam_right": 2.0, "cam_side": 0.9, "cam_top": 0.8}
--policy.learnable_camera_scales=false
--policy.use_camera_id_embed=true
```

---

## 6. 安装与使用

### 6.1 前置条件

- 已安装并激活 LeRobot Python 环境（含 `lerobot-train`）
- 数据集已录制完成（`data/lerobot/local/cube_v4_3view_side/`）

### 6.2 安装插件

在 LeRobot 环境中执行（**不要**重复拉取 torch/lerobot 依赖）：

```bash
cd piper
pip install -e . --no-deps
```

验证注册：

```bash
python -c "import lerobot_policy_act_piper; from lerobot.configs.policies import PreTrainedConfig; print('act_piper' in PreTrainedConfig.get_known_choices())"
# 应输出 True
```

### 6.3 训练

```bash
# 预览命令
python -m piper_towel_fold.start_training --config configs/record_pick_cube.json --dry-run

# 正式训练
bash scripts/start_training.sh configs/record_pick_cube.json
```

输出 checkpoint：

```
outputs/train/act_piper_cube_v4_3view_side/checkpoints/{step}/pretrained_model/
```

### 6.4 推理

推理脚本**无需修改**。确保：

1. 当前环境已 `pip install -e . --no-deps`
2. `policy_live.policy_path` 指向 `act_piper` 训练的 checkpoint
3. 预处理配置与训练数据一致（参见 `docs/data-preprocessing.md`）

```bash
bash scripts/run_policy_live.sh configs/record_pick_cube.json
```

---

## 7. 消融实验建议

每次实验修改 `training.job_name` / `training.output_dir`，避免覆盖：

| 实验 | `policy_type` | `camera_scales` | `use_camera_id_embed` | 目的 |
|------|---------------|-----------------|----------------------|------|
| A0 baseline | `act` | — | — | 原版对照 |
| A1 仅缩放 | `act_piper` | right=2.0, top=0.8 | `false` | 验证 α 单独效果 |
| A2 仅 embed | `act_piper` | 全 1.0 或删除字段 | `true` | 验证身份 embedding |
| **A3 组合** | `act_piper` | right=2.0, top=0.8 | `true` | **当前默认，主推** |

评估指标：

- 真机夹取成功率 / 偏移量
- `tools/check_policy_vision.py`：分别黑掉 cam_right / cam_top，观察策略敏感度变化
- `offline_infer` 离线 action MAE

α 扫描（固定 embed 开启）：

```
cam_right ∈ {1.5, 2.0, 2.5}
cam_top   ∈ {0.6, 0.8, 1.0}
```

---

## 8. 注意事项

### 8.1 Checkpoint 兼容性

- **vanilla ACT checkpoint 不能加载到 `act_piper`**（多了 `camera_scales` buffer 和 `camera_id_embed` 参数）
- **必须从头训练**；旧 `act_cube_v4_3view_side` checkpoint 仅适用于 `policy.type=act`

### 8.2 相机顺序

缩放系数按 `config.image_features` 枚举顺序索引，该顺序来自数据集 `meta/info.json`（通常字母序：`cam_right` → `cam_side` → `cam_top`）。

`camera_scales` 用**短相机名**作 key，与 `observation.images.*` 前缀无关，换相机配置时更安全。

### 8.3 与预处理的关系

- 图像色彩增强（`preprocessing.py`）在录制/离线阶段完成，与本次模型改动**正交、可并存**
- 训练时 LeRobot 仍不做随机 augment；差异化 per-camera 增强需另做（见 `docs/data-preprocessing.md` §11）

### 8.4 LeRobot 版本

插件继承 `lerobot.policies.act.modeling_act.ACT`，`ACTPiper.forward` 与上游 ACT 保持同步。LeRobot 大版本升级后若 `modeling_act.py` 有变，需检查 `modeling_act_piper.py` 是否需同步。

---

## 9. 后续可扩展方向（未实现）

| 方向 | 说明 | 前置条件 |
|------|------|----------|
| **StateGate 动态门控** | MLP 根据关节/夹爪/距离输出每相机权重 | 需离线计算末端—方块距离 |
| **Attention mask（3B）** | 近距离限制俯视对腕相的注意力 | 需改 `ACTEncoderLayer`，侵入性高 |
| **辅助 bbox loss** | 腕相图像上方块相对偏移监督 | 需标注或伪标签 pipeline |
| **可学习 α + 熵正则** | `learnable_camera_scales=true` 防退化 | 配置开关即可，待实验 |
| **训练时随机增强** | 腕相小扰动 / 俯视大扰动 | 需 fork LeRobot Dataset transform |

---

## 10. 相关文件索引

```
piper/
├── pyproject.toml                              # lerobot_policy_act_piper 包定义
├── configs/record_pick_cube.json               # 训练/推理统一配置（已切 act_piper）
├── docs/
│   ├── act-piper-camera-fusion.md              # 本文档
│   └── data-preprocessing.md                   # 预处理说明（独立）
├── src/
│   ├── lerobot_policy_act_piper/
│   │   ├── __init__.py
│   │   ├── configuration_act_piper.py          # ACTPiperConfig
│   │   ├── modeling_act_piper.py               # ACTPiper / ACTPiperPolicy
│   │   └── processor_act_piper.py
│   └── piper_towel_fold/
│       └── start_training.py                   # 透传 act_piper 超参
└── scripts/start_training.sh                   # 训练入口
```

---

*文档版本：与 `lerobot_policy_act_piper` v0.1.0 初始实现同步。*
