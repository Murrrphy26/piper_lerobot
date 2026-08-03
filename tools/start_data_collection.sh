#!/bin/bash

echo "========================================="
echo "数据采集系统启动"
echo "========================================="
echo ""
echo "请选择采集模式:"
echo "  1. 仅Orbbec测试 (验证深度/IR)"
echo "  2. 所有摄像头采集 (推荐)"
echo "  3. 自定义采集"
echo ""
read -p "选择 (1/2/3): " mode

case $mode in
    1)
        echo "启动Orbbec测试..."
        python3 /home/agilex/code/yjw/cameras/orbbec_gemini_config.py
        ;;
    2)
        echo "启动所有摄像头采集..."
        python3 /home/agilex/code/yjw/cameras/all_cameras_collector.py
        ;;
    3)
        echo "输入自定义Python脚本路径:"
        read script_path
        python3 $script_path
        ;;
    *)
        echo "无效选择，默认启动所有摄像头采集"
        python3 /home/agilex/code/yjw/cameras/all_cameras_collector.py
        ;;
esac
