# Piper六轴机械臂PI05操作手册

## 前置安全注意事项

若机械臂从臂锁死无法活动，复原操作步骤（务必注意安全）：

1. 用手托稳机械臂本体，再拔下从臂航空插头；

2. 重新插接主臂航空头，等待机械臂复位。

# 快速开始

## 1. 打开工控机CAN总线初始化

```bash
cd ~/code/yjw/piper
conda activate piper
bash scripts/bringup_can.sh
```

## 2. 本地录制数据集

1. 硬件准备：接好四台机械臂航空接头，搭建任务操作环境；

2. 配置修改：打开配置文件，修改本次任务名称、数据存储路径；

3. 采集要求：

- 录制时方块摆放位置尽量随机，提升数据泛化性；

- 全程操作平稳可控，所有抓取动作**分步停顿执行**，禁止连贯快速操作，标准流程：先定位到方块上方 → 停顿稳定 → 打开夹爪 → 停顿稳定 → 向下落位 → 停顿稳定 → 合拢夹爪 → 停顿稳定，完成单次抓取后再进行后续动作；

启动录制脚本（默认配置为橙色方块抓取任务）：

```bash
bash scripts/start_recording_pi05.sh
```

等待终端打印第二遍 `Press Ctrl C` 即可开始操作。

单次采集结束后输入数据标签：

- `s`：任务执行成功，数据参与后续训练

- `f`：任务执行失败，数据丢弃不参与训练

禁止手动删除本地已录制数据集。

录制全部完成后，打包上传数据集至远程服务器：

```bash
bash scripts/upload_dataset_to_remote.sh
```

## 3. 远程服务器模型训练

网络波动较大，推荐使用tmux会话保持训练进程：

```bash
ssh allinai2
cd /mnt/disk/fyx/piper
conda activate piper
```

项目根目录：`/mnt/disk/fyx/piper`

修改配置文件：调整任务名称、日志存储路径、训练迭代步数等超参。

启动训练脚本：

```bash
bash scripts/start_training_pi05.sh
```

## 4. 远程异步推理（严格按顺序执行，禁止跳步）

### 步骤1：远程GPU服务端启动Policy Server（终端持续保持运行）

```bash
ssh allinai2
cd /mnt/disk/fyx/piper
bash scripts/start_policy_server_pi05.sh
```

**必须等待以下完整日志输出，才可执行后续步骤**：

```bash
INFO ... Preloading policy at startup | type=pi05 | path=outputs/train/...
INFO ... Startup preload finished in XXXs
INFO ... PolicyServer started on 0.0.0.0:8080
```

说明：开启开机预加载模型配置（`policy_server.preload_at_startup: true`），首次启动加载耗时1–3分钟，加载完成服务才监听8080端口。

### 步骤2：工控机新开终端，建立SSH隧道（终端不可关闭）

```bash
cd ~/code/yjw/piper
conda activate piper
bash scripts/ssh_tunnel_policy_server.sh
```

正常成功输出示例：

```bash
Opening SSH tunnel for policy server gRPC.
  local:  127.0.0.1:8080
  remote: 127.0.0.1:8080
  ssh host: allinai2
```

关闭该终端会直接断开推理通信链路。

### 步骤3：工控机新开终端，启动机器人推理客户端

```bash
cd ~/code/yjw/piper
conda activate piper
```

推理前手动初始化CAN总线与机械臂（客户端脚本不会自动初始化硬件）：

```bash
bash scripts/bringup_can.sh
# bash scripts/reset_arms.sh   # 如需退出示教模式取消注释执行
```

启动远程异步推理客户端：

```bash
bash scripts/run_async_policy_client_pi05_remote.sh
```

# 常见问题排查

## 一、数据录制阶段

1. 执行 `scripts/start_policy_server_pi05.sh` 报OpenCV相关报错

原因：相机图像流存在延迟冲突；解决方案：重复执行一次脚本即可恢复。

## 二、异步推理阶段

1. 网络延迟过高导致推理卡顿、机械臂动作滞后

排查命令（在工控机执行）：

```bash
ssh allinai2 "ping 127.0.0.1 -c 10"
```

通过ping时延判断服务器与工控机通信质量，时延过高可切换有线网络或重启SSH隧道。
