# Piper Plan

这个计划文档只关注“从当前能读关节角度”推进到“能做第一轮模仿学习采集”的近期工作，不替代长期路线书。

## 当前状态

已完成：

- `PiperRobotConfig` 基本定义完成
- `PiperRobot` 可以连接左右从臂
- `PiperRobot` 可选连接左右主臂，并读取主臂 action
- 可以读取左右臂 6 个关节角和夹爪状态
- 有独立测试脚本 `tools/test_read_state.py`
- 已实测确认 CAN 映射：左臂 `can2`，右臂 `can0`
- 已补充 CAN 拉起脚本和状态读取启动脚本
- 已确认主臂控制从臂功能良好
- 已确定第一个练手任务为“抓方块”
- 已新增 LeRobot episode recorder，并保留 JSONL 调试 recorder
- 已新增摄像头探测脚本和图片落盘格式
- 已实测三摄映射：`2 -> cam_top`，`4 -> cam_left`，`0 -> cam_right`

未完成：

- 动作下发
- 数据集加载检查、任务标注、训练配置

## 近期目标

### Milestone 1: 打通状态读取

目标：

- 在工控机上稳定读取双臂关节角
- 确认左右臂 CAN 口映射正确
- 确认单位和刷新周期正确

验收标准：

- 运行 `PYTHONPATH=src python tools/test_read_state.py --left-can <left> --right-can <right>` 成功
- 左右各 6 个关节角随手动运动实时变化
- 左右夹爪状态能变化
- 连续运行 10 分钟不中断

待办：

1. 确认读数单位是否符合预期。
2. 记录每个关节的零位或静止位读数。
3. 确认最合适的打印或采样频率。
4. 记录一份上机 SOP。

### Milestone 2: 最小控制闭环

目标：

- 从“只读状态”进入“可控”
- 明确后续 teleop 和录制接口的动作空间

验收标准：

- 至少可以向单臂或双臂发送一组简单动作
- 有急停和安全边界
- 能验证动作命令和反馈状态一致

待办：

1. 在 `PiperRobot.send_action()` 中补最小可用实现。
2. 确定动作空间是“关节位置”还是“末端增量”。
3. 定义左右夹爪动作范围。
4. 加入基本安全检查。

### Milestone 3: 主从示教

目标：

- 用主臂带从臂完成基础示教
- 为后续采集提供稳定的 teleop 方式

验收标准：

- 主臂运动可稳定映射到从臂：已确认
- 双臂示教时延可接受：已确认
- 夹爪同步可用：待实机持续验证

待办：

1. 明确主臂是否接入工控机 CAN；如果没有，先用 follower 状态作为 action smoke test。
2. 验证左右臂同步和夹爪映射。
3. 录 5 到 10 条 LeRobot 短 episode，检查 action 和 observation 是否连续。
4. 测试长时间运行稳定性。

### Milestone 3.5: 最小试录

目标：

- 先录“抓方块”的短 episode
- 直接写成 `LeRobotDataset`
- 验证 episode 文件、时间戳、action、observation 和图像字段

验收标准：

- `PYTHONPATH=src python -m piper_towel_fold.record_episode --dataset-format lerobot --task pick_cube --action-source follower --duration 30` 可成功生成 dataset
- `data/lerobot/local/piper_pick_cube/` 下有 `data/`、`meta/`、`videos/`
- `observation.state` 和 `action` 都是 14 维
- 录制过程可用 `Ctrl+C` 安全停止
- 手动停止后可用 `--prompt-outcome` 标记成功、失败或未知

待办：

1. 用 follower action 录一条 30 秒 smoke test。
2. 如果主臂 CAN 可读，改用 leader action 再录一条。
3. 检查帧率是否接近目标 fps。
4. 记录每条 episode 的成功/失败和异常现象。

### Milestone 4: 相机与观测同步

目标：

- 加入 3 路摄像头
- 让机器人状态和视觉观测同时进入同一采集流程

验收标准：

- `python tools/test_cameras.py --indices 0,1,2,3,4,5` 能找到 3 路可用画面
- `python -m piper_towel_fold.record_episode --camera-indices 2,4,0 --camera-names cam_top,cam_left,cam_right` 能同时保存 3 路视频
- LeRobot dataset 中存在 `observation.images.cam_top`、`observation.images.cam_left`、`observation.images.cam_right`
- 机械臂状态、action 和图像能同时采样
- 数据字段命名稳定，不再频繁变动

待办：

1. 确认 3 个摄像头编号和视角命名。
2. 录一条 30 秒三摄 smoke test。
3. 用 `LeRobotDataset` 加载检查帧数、episode 数和图像字段。
4. 评估视频编码是否影响目标 fps。
5. 固化三摄视角命名。

### Milestone 5: 第一轮 LeRobot 采集

目标：

- 先做一个简单任务跑通完整流程
- 叠毛巾之前，可以先用“捡方块”熟悉记录、训练、回放

建议任务顺序：

1. 单步抓取方块
2. 双臂协同搬运方块
3. 受控场景下的半折毛巾

验收标准：

- 成功录到第一版数据集
- 至少完成一次训练和一次回放
- 明确一份可复用 SOP

待办：

1. 先选定第一个练手任务，建议优先“捡方块”。
2. 定义 episode 开始/结束条件。
3. 记录成功/失败标签。
4. 跑一次最小训练闭环。

## 对“下一步是不是采集数据”的判断

结论：

- 可以开始最小试采集
- 还不建议立刻大规模正式采集

原因：

- 主从示教已经可用，可以先录“抓方块”验证数据链路
- 当前还没做数据集加载和可视化检查
- 图像观测需要实机确认帧率和编码稳定性
- 叠毛巾任务本身较复杂，直接采容易把问题混在一起

建议的最近两步：

1. 用 `python -m piper_towel_fold.record_episode` 录 5 到 10 条抓方块短 episode。
2. 用三摄像头直接录制一条 LeRobot smoke test，并加载检查。

## 推荐执行顺序

1. 先完成 Milestone 1，别急着进训练。
2. 先把 CAN 启动和状态读取流程脚本化。
3. 用主从示教录最小 LeRobot episode。
4. 完成 Milestone 4，把相机接入采集闭环。
5. 做 dataset 加载、可视化和训练配置。
6. 再做正式训练和回放。

## 风险点

- `lerobot` 和 `piper_sdk` 在工控机环境中的版本兼容性
- 双臂主从时延导致示教不稳定
- CAN 口映射或设备命名不固定
- 当前仓库还不是完整插件结构，后续接 `LeRobot` CLI 时可能需要重构目录
- 叠毛巾任务难度高，直接上手容易在数据质量阶段卡住

## 这周建议产出

如果你想先快速推进一周，我建议本周只盯这四件事：

1. 在工控机上把双臂状态读取跑稳定。
2. 固化 CAN 映射、启动命令和实测现象。
3. 录 5 到 10 条“抓方块”短 episode。
4. 明确主臂 action 是否能由工控机直接读取。
