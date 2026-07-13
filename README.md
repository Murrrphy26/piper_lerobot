# 快速开始 (Pi0.5)

## 1. 打开工控机

```bash
cd code/yjw/piper
conda activate piper
bash scripts/bringup_can.sh
```

## 2. 录制数据

连接四个机械臂的航空头，布置好任务环境后  
在config里修改任务名称以及存储路径

```bash
bash scripts/start_recording_pi05.sh #当前的config配置是抓取橙色方块
```

在每录制完一次数据之后会有一个标签（s/f/u）:若录制的任务成功输入s即可，若任务失败输入f（不会参与训练），请不要手动删除已录制的数据

录制完成后将数据打包发送到远程服务器

## 3. 远程训练

连接远程服务器

```bash
ssh allinai2
cd /mnt/disk/fyx/piper
conda activate piper
```

项目文件在/mnt/disk/fyx/piper  
将数据解压到目标路径  
在config里修改任务名称、存储路径以及训练步数等

开始训练

```bash
bash scripts/start_training_pi05.sh
```

## 4. 远程异步推理

必须按顺序执行，**不可跳步**。

### 步骤 1：在远程 GPU 服务器启动 Policy Server（保持终端不关）

```bash
ssh allinai2
cd /mnt/disk/fyx/piper
bash scripts/start_policy_server_pi05.sh
```

**等待出现以下日志后再进行下一步：**

```
INFO ... Preloading policy at startup | type=pi05 | path=outputs/train/...
INFO ... Startup preload finished in XXXs
INFO ... PolicyServer started on 0.0.0.0:8080
```

> 首次启动预加载约 **1–3 分钟**（`policy_server.preload_at_startup: true`）。  
> 预加载完成后 Server 才开始监听 8080。

### 步骤 2：在工控机开启 SSH 隧道（保持终端不关）

新开终端：

```bash
cd code/yjw/piper
conda activate piper
bash scripts/ssh_tunnel_policy_server.sh
```

正常输出示例：

```
Opening SSH tunnel for policy server gRPC.
  local:  127.0.0.1:8080
  remote: 127.0.0.1:8080
  ssh host: allinai2
```

**此终端必须保持运行**，关闭即断连。

### 步骤 3：在工控机启动 Robot Client

再开一个终端：

```bash
cd code/yjw/piper
conda activate piper
bash scripts/run_async_policy_client_pi05_remote.sh
```