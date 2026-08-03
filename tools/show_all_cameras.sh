#!/bin/bash

echo "========================================="
echo "所有摄像头设备信息"
echo "========================================="

echo -e "\n1. V4L2设备列表:"
echo "-----------------------------------------"
v4l2-ctl --list-devices

echo -e "\n2. Video设备节点:"
echo "-----------------------------------------"
ls -l /dev/video* 2>/dev/null | awk '{print $9, $10, $11}'

echo -e "\n3. Media设备节点:"
echo "-----------------------------------------"
ls -l /dev/media* 2>/dev/null

echo -e "\n4. 每个video设备的详细信息:"
echo "-----------------------------------------"
for dev in /dev/video*; do
    if [ -e "$dev" ]; then
        echo -e "\n$dev:"
        v4l2-ctl -d "$dev" --all 2>/dev/null | grep -E "Driver|Card|Bus|Width|Height|Pixel Format" | head -5
    fi
done

echo -e "\n5. USB设备信息:"
echo "-----------------------------------------"
lsusb | grep -i "camera\|orbbec\|piper"

echo -e "\n6. udev设备信息:"
echo "-----------------------------------------"
for dev in /dev/video*; do
    if [ -e "$dev" ]; then
        echo -e "\n$dev:"
        udevadm info --query=all --name="$dev" 2>/dev/null | grep -E "ID_VENDOR|ID_MODEL|ID_SERIAL|ID_PATH" | head -3
    fi
done

echo -e "\n========================================="
