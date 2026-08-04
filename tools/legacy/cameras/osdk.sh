#!/bin/bash
# 使用Orbbec SDK的HelloOrbbec示例拍照

echo "========================================="
echo "使用Orbbec SDK拍照"
echo "========================================="

# 设置环境变量
export LD_LIBRARY_PATH=/home/agilex/orbbec_ws/src/OrbbecSDK/lib/linux_x64:$LD_LIBRARY_PATH

# 进入示例目录
cd /home/agilex/orbbec_ws/src/OrbbecSDK/build

# 如果有编译好的示例
if [ -f "./bin/OBHelloOrbbec" ]; then
    echo "运行 HelloOrbbec 示例..."
    ./bin/OBHelloOrbbec
elif [ -f "./bin/ob_hello_orbbec" ]; then
    echo "运行 ob_hello_orbbec 示例..."
    ./bin/ob_hello_orbbec
else
    echo "示例程序未编译，正在编译..."
    cd /home/agilex/orbbec_ws/src/OrbbecSDK
    mkdir -p build && cd build
    cmake ..
    make -j4
    ./bin/OBHelloOrbbec
fi