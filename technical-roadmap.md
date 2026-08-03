# Piper + LeRobot 叠毛巾技术路线书

## 1. 项目目标

在 `Piper` 机械臂平台上接入 `LeRobot` 工作流，完成以下闭环：

1. 真实机器人数据采集
2. LeRobot 格式数据集构建与管理
3. 模仿学习训练与离线评估
4. 在线部署、纠错采集与迭代训练
5. 最终实现“叠毛巾”任务的稳定执行

本路线默认你当前平台是**双臂 Piper 主从臂 + 双夹爪 + 桌面相机**。其中“主从臂”指一套可用于示教/遥操作的 leader-follower 结构，目标是在 LeRobot 中完成双臂数据采集、训练、部署与迭代。

## 2. 先给结论

这个项目的关键不在“训练一次模型”，而在先把下面三件事做扎实：

1. **把 Piper 封装成 LeRobot 可识别的 robot plugin**
2. **把叠毛巾拆成可采集、可评估的子任务**
3. **建立 DAgger 式迭代闭环，而不是只靠一次示教数据**

如果直接以“完整叠毛巾 end-to-end”开局，双臂平台虽然明显优于单臂，但仍会卡在：

- 布料初始形态分布太大
- 双臂时序配合和对边精度要求高
- 失败恢复难，离线示教覆盖不够

所以更稳妥的路线是：**先用双臂完成结构化场景下的完整对折/双边对齐，再向完整自由摆放毛巾推进。**

## 3. 可行性判断

### 3.1 LeRobot 侧

LeRobot 官方文档当前已经覆盖了真实机器人数据采集、训练、评估和 human-in-the-loop 纠错闭环。官方文档显示：

- 可以用 `python -m lerobot.record` 录制真实机器人数据集
- 可以用 `python lerobot/scripts/train.py` 训练策略
- 可以用 `lerobot-rollout` 进行真实机器人部署
- 可以用 `--strategy.type=dagger` 做人类介入纠错采集

如果硬件不是 LeRobot 内置支持型号，官方推荐走 **Bring Your Own Hardware** 路线，实现自定义 `Robot` 类并做成可安装插件包。

### 3.2 Piper 侧

AgileX 官方 `piper_sdk` 当前提供 Python SDK，仓库说明它用于接收和处理机械臂 CAN 数据帧，并支持通过 `pip3 install piper_sdk` 安装。官方说明还给出了 CAN 工具链依赖和配置方式。

### 3.3 任务侧

叠毛巾属于**可变形物体操作**。这一类任务的核心难点不是轨迹控制，而是：

- 视觉状态估计
- 角点/边缘可见性
- 接触后的布料形变不确定性
- 失败恢复

因此项目的决定因素不是“Piper 能不能动起来”，而是：

1. 相机视角是否稳定覆盖布面
2. 数据是否覆盖关键布料形态
3. 子任务拆分是否合理
4. 是否支持在线纠错再采集

## 4. 总体技术路线

建议按四阶段推进。

### 阶段 A：基础设施打通

目标：让双臂 Piper 主从系统进入 LeRobot 生态。

工作项：

1. 建立硬件控制链路
   - 验证 `piper_sdk` 单独可控
   - 打通左右臂 CAN 通信、回零、夹爪控制、急停
   - 明确 leader arm 到 follower arm 的映射关系、频率和延迟
2. 建立相机链路
   - 至少 1 路顶视相机
   - 推荐增加 1 路斜侧视相机
3. 开发 LeRobot robot plugin
   - 将 Piper 封装为 `lerobot_robot_piper`
   - 实现 observation / action / connect / disconnect / calibrate / configure
   - 明确双臂命名、同步时钟和双夹爪状态
4. 定义任务动作空间
   - 首版优先用双臂关节位置控制或双末端位姿增量控制
   - 左右夹爪开合独立成 action 维度

交付物：

- `lerobot_robot_piper` 可安装包
- `piper` 在 LeRobot CLI 下可被识别
- 双臂 `teleoperate / record / rollout` 冒烟通过

### 阶段 B：任务分解与首轮示教采集

目标：先做“双臂能稳定复现”的子任务，而不是直接赌开放场景完整任务。

建议子任务拆分：

1. `dual_corner_grasp`
   - 双臂分别抓取两个目标角点
2. `edge_lift_and_tension`
   - 双臂提边并拉紧，建立稳定布料张力
3. `edge_alignment`
   - 双臂将当前边对齐到目标边
4. `half_fold_bimanual`
   - 双臂完成一次对折
5. `place_and_smooth`
   - 放下后双臂协同抚平
6. `full_towel_fold_bimanual`
   - 双臂完成完整叠放

路线原则：

- 第一阶段先做 `half_fold_bimanual`
- 初始布料摆位尽量受控
- 先固定毛巾尺寸、颜色、桌面材质、光照
- 主从示教动作优先保证双臂同步和张力稳定，不追求速度

交付物：

- 子任务定义文档
- 成功判定标准
- 采集 SOP

### 阶段 C：训练、评估、纠错迭代

目标：让数据闭环跑起来。

工作项：

1. 首轮示教采集
   - 每个子任务先采 50 到 200 条 episode
   - 记录成功/失败标签
2. 首轮策略训练
   - 优先从 `ACT` 起步
   - 后续再比较 `SmolVLA / Pi0.5 / X-VLA` 等视觉语言动作模型
3. 真实机评估
   - 固定测试集场景
   - 统计成功率、平均耗时、失败类型
4. DAgger 纠错采集
   - 在策略执行中人工接管
   - 重点补双臂抓取失配、拉偏、边缘错位附近的数据
5. 迭代训练
   - 数据版本化
   - 比较 `v1 / v2 / v3` 成功率变化

交付物：

- `dataset_v1/v2/v3`
- `policy_half_fold_bimanual_v1/v2/v3`
- 评估报告

### 阶段 D：从受控场景走向完整叠毛巾

目标：从实验可行走向任务可用。

推进顺序建议：

1. 固定初始摆放的双臂对折毛巾
2. 轻微扰动下的双臂对折毛巾
3. 不同颜色同尺寸毛巾
4. 不同厚度毛巾
5. 部分褶皱毛巾
6. 完整自由摆放毛巾

如果双臂在 `full_towel_fold_bimanual` 上长期失败，建议引入以下之一：

- 桌面治具：定位边框、摩擦分区、压条
- 更适合布料的夹爪或吸附末端
- 双臂动作时序约束或显式阶段控制
- 视觉前处理：角点/边缘检测辅助

## 5. 系统架构建议

建议采用下面的最小可行架构：

1. **机器人控制层**
   - `piper_sdk`
   - CAN 接口
   - 左右臂与双夹爪控制

2. **LeRobot 设备适配层**
   - `PiperRobotConfig`
   - `PiperRobot`
   - leader-follower 映射
   - 相机封装
   - 标定与配置管理

3. **数据采集层**
   - `lerobot.record`
   - 任务标签
   - episode 元数据

4. **训练层**
   - LeRobot train pipeline
   - 实验配置管理
   - checkpoint 管理

5. **部署与评估层**
   - `lerobot-rollout`
   - 自动评估记录
   - DAgger 人类纠错
   - 双臂同步监控

## 6. 关键工程设计

### 6.1 Robot Plugin 设计

这是整个项目的第一关键路径。

按 LeRobot 官方 BYOH 机制，你需要单独做一个可安装 Python 包，包名建议：

- `lerobot_robot_piper`

核心类建议：

- `PiperRobotConfig`
- `PiperRobot`
- `PiperTeleopConfig`
- `PiperTeleop`

至少实现：

- `observation_features`
- `action_features`
- `connect()`
- `disconnect()`
- `is_connected`
- `get_observation()`
- `send_action()`
- `configure()`
- `calibrate()` 或空实现

建议首版 observation：

- `left_arm.joint_1.pos` ... `left_arm.joint_6.pos`
- `right_arm.joint_1.pos` ... `right_arm.joint_6.pos`
- `left_gripper.pos`
- `right_gripper.pos`
- `camera.top`
- `camera.side`

建议首版 action：

- 左臂 6 维关节目标
- 右臂 6 维关节目标
- 左右夹爪各 1 维开合

建议首版 teleop observation / action：

- leader 左臂关节状态
- leader 右臂关节状态
- 夹爪状态

原因是双臂主从系统的第一关键不只是 robot plugin，还包括 **teleop plugin**。没有稳定双臂示教，后面的数据质量不会好。

### 6.2 控制方式选择

首版建议优先级：

1. 双臂关节位置控制
2. 双末端位姿增量控制
3. 力控/阻抗控

对这个项目，第一版不建议直接上复杂力控。原因是你当前的主要瓶颈更可能是：

- 数据不足
- 视觉不稳定
- 双臂同步误差和张力控制不稳定

而不是控制器不够先进。

### 6.3 视觉方案

最低配置：

- 1 路顶视 RGB 相机

推荐配置：

- 1 路顶视 RGB
- 1 路斜侧视 RGB

如果预算允许，再考虑：

- 深度相机

视觉重点不是“多传感器越多越好”，而是：

1. 布面边界清晰
2. 角点无遮挡
3. 光照稳定
4. 背景对比强

对于毛巾任务，桌面和毛巾颜色一定要拉开。

### 6.4 任务表示

建议任务文本从一开始就固定下来，便于后续兼容视觉语言动作模型。

例如：

- `Fold the towel in half.`
- `Grasp the near-left corner of the towel.`
- `Align the lifted edge with the opposite edge.`

如果你只训练单任务策略，也建议保留 `task text` 字段，后续扩展成本更低。

## 7. 数据采集路线

### 7.1 采集原则

双臂叠毛巾不是靠“大而杂”数据取胜，而是靠“覆盖关键状态和关键协同模式”。

优先采这些状态：

1. 平整且易抓的初始状态
2. 轻微褶皱状态
3. 左右臂抓点轻微不对称但可恢复的状态
4. 边缘拉紧、对齐、放下前后的过渡状态
5. 失败恢复状态

### 7.2 episode 设计

建议一开始每个 episode 只对应一个清晰子任务。

例如：

- `half_fold_bimanual`: 从已铺平毛巾开始，到双臂完成一次对折结束

不要把下面这些混在一个 episode：

- 找毛巾
- 铺平
- 抓角
- 对折
- 压实

这样会让策略学习目标过长、失败诊断困难。

### 7.3 首轮数据规模建议

面向单子任务 `half_fold_bimanual` 的保守建议：

- 首轮示教：100 到 300 条 episode
- 每条时长：10 到 30 秒
- 相机：30 FPS

如果成功率很低，不要先加到 1000 条，先看失败分布：

- 是左右臂抓点不一致？
- 是提边后张力不足或过大？
- 是边缘对齐误差大？
- 是放下阶段双臂释放时序不合适？

数据应该按失败模式补，而不是盲目堆量。

### 7.4 在线纠错采集

LeRobot 当前支持 `lerobot-rollout --strategy.type=dagger` 的 human-in-the-loop 采集。对布料任务，这一步非常关键。

建议节奏：

1. 用离线示教训练首版策略
2. 上机运行
3. 人工只在即将失败时接管纠正
4. 合并纠错数据重新训练

这个闭环通常比单纯继续录示教更有效。

## 8. 训练路线

### 8.1 第一阶段策略

建议：

- **先用 ACT 打双臂基线**

原因：

- LeRobot 官方真实机教程直接给出 ACT 训练流程
- ACT 对中等规模示教数据更容易先跑出结果
- 你当前第一目标是验证系统闭环，不是追求最强 foundation policy

### 8.2 第二阶段策略

当以下条件满足后，再考虑更强模型：

1. Piper + LeRobot 已稳定
2. `half_fold_bimanual` 已有可复现实验
3. 数据集达到可观规模

再评估：

- `SmolVLA`
- `Pi0.5`
- `X-VLA`

如果后续希望加入任务文本、多任务泛化、跨场景泛化，这类 VLA 模型更值得投入。

### 8.3 训练配置管理

必须从第一天开始做的不是“调模型”，而是：

- 数据集版本号
- 相机配置版本号
- 机械臂固件/SDK 版本号
- 训练配置版本号
- 评测场景版本号

否则很快会出现“模型变好了还是环境变简单了”这种不可追溯问题。

## 9. 评估体系

### 9.1 成功标准

先定硬标准，再训练。

建议 `half_fold_bimanual` 成功定义：

1. 双臂完成一次对折
2. 折边偏差小于设定阈值
3. 双臂释放后毛巾保持目标折叠形态
4. 在时限内完成

### 9.2 指标

至少记录：

- 任务成功率
- 平均完成时间
- 抓取成功率
- 对齐误差
- 失败模式分布
- 人工接管次数

### 9.3 测试集分层

测试不要只测训练分布。

建议三层测试集：

1. `easy`
   - 固定摆放、固定毛巾、固定光照
2. `medium`
   - 小扰动、轻微褶皱
3. `hard`
   - 初始姿态变化更大、边角翻卷

只有 `easy` 很高成功率后，才值得进入下一层。

## 10. 现实风险与应对

### 风险 1：双臂主从映射不稳定

表现：

- 双臂示教时 follower 跟随误差大，采集轨迹抖动，左右臂不同步

应对：

- 先把主从链路单独打稳，再进入 LeRobot
- 固定采集频率和同步机制
- 必要时在示教端做插值、限速、滤波

### 风险 2：示教方式不顺手

表现：

- 双臂示教负担高，人工长时间操作容易引入不一致

应对：

- 优先把双臂 leader arm 方案标准化
- 将任务分段采集，降低单条 episode 长度
- 允许先用更慢但一致性更高的采集节奏

### 风险 3：视觉泛化差

表现：

- 换一条毛巾或光照后成功率明显下跌

应对：

- 先固定背景和照明
- 逐步扩展颜色、材质、摆放扰动
- 先做课程式泛化，不要一次性全放开

### 风险 4：训练有效，部署失败

表现：

- 离线看起来正常，真机执行发散

应对：

- 核查 observation/action 时间同步
- 核查相机延迟
- 核查动作频率与执行周期
- 用短时 rollout + DAgger 修正分布偏移

## 11. 推荐实施计划

### 第 1-2 周：系统打通

- 跑通 `piper_sdk`
- 确认相机输入
- 完成 `lerobot_robot_piper` 骨架
- 实现基本 observation/action 接口

里程碑：

- LeRobot 能连接 Piper 并读取状态、发送动作

### 第 3-4 周：采集闭环

- 实现双臂 teleop 方案
- 录制 `half_fold_bimanual` 首批数据
- 建立评估脚本与标注规范

里程碑：

- 采集 100+ 条可用 episode

### 第 5-6 周：首轮训练与真机验证

- ACT 基线训练
- 真机 rollout
- 统计失败模式

里程碑：

- 在 `easy` 测试集上出现双臂可重复成功案例

### 第 7-8 周：DAgger 迭代

- 收集失败附近纠错数据
- 二轮训练
- 增强轻微扰动泛化

里程碑：

- `easy` 成功率稳定
- `medium` 开始可用

### 第 9 周以后：扩到完整叠毛巾

- 从 `half_fold_bimanual` 扩展到 `full_towel_fold_bimanual`
- 评估是否需要治具或新夹爪

## 12. 项目目录建议

建议在 `piper` 下按下面组织：

```text
piper/
├── technical-roadmap.md
├── docs/
│   ├── task-definition.md
│   ├── data-collection-sop.md
│   ├── evaluation-protocol.md
│   └── risk-register.md
├── lerobot_robot_piper/
│   ├── pyproject.toml
│   ├── src/lerobot_robot_piper/
│   │   ├── __init__.py
│   │   ├── piper_config.py
│   │   ├── piper_robot.py
│   │   └── cameras.py
├── scripts/
│   ├── teleop_piper.sh
│   ├── record_half_fold_bimanual.sh
│   ├── train_act_half_fold_bimanual.sh
│   ├── rollout_half_fold_bimanual.sh
│   └── dagger_half_fold_bimanual.sh
└── experiments/
    ├── dataset_versions.md
    └── model_results.md
```

## 13. 第一优先级待办

按优先级排序，你现在最该做的是：

1. **确认你的 Piper 控制接口边界**
   - 关节读写频率
   - 夹爪控制方式
   - 主从臂映射、是否有 teach/drag 模式
2. **确定 teleop 方案**
   - 双 leader arm / 自定义双臂示教
3. **确定相机方案**
   - 顶视为主，保证布面完整可见
4. **先把任务缩到 `half_fold_bimanual`**
5. **同时开发 `lerobot_robot_piper` 和 `lerobot_teleoperator_piper`，并跑通一次 `teleoperate -> record -> train -> rollout`**

没有这五步，后面讨论模型选型基本都太早。

## 14. 我对这个项目的判断

如果你的目标是：

- 双臂主从 Piper
- 有稳定双臂示教
- 顶视相机为主、侧视为辅
- 固定桌面与光照
- 固定毛巾尺寸
- 先做 `half_fold_bimanual`
- 通过 DAgger 持续补双臂失配和对边失败数据

那么这个项目是有明确落地路径的。

## 15. 参考依据

1. LeRobot 官方文档首页：
   https://huggingface.co/docs/lerobot/main/index
2. LeRobot 官方真实机器人入门：
   https://huggingface.co/docs/lerobot/main/getting_started_real_world_robot
3. LeRobot 官方部署与 DAgger：
   https://huggingface.co/docs/lerobot/main/inference
4. LeRobot 官方 Bring Your Own Hardware：
   https://huggingface.co/docs/lerobot/main/integrate_hardware
5. AgileX 官方 `piper_sdk`：
   https://github.com/agilexrobotics/piper_sdk
6. AgileX 官方 PiPER 产品页：
   https://global.agilex.ai/products/piper
7. LeRobot 官方衣物折叠参考空间：
   https://huggingface.co/spaces/lerobot/robot-folding
