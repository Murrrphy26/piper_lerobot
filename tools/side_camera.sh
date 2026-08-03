#!/bin/bash

# 创建照片目录
PHOTO_DIR=photos
mkdir -p "$PHOTO_DIR"

# 获取时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================="
echo "所有摄像头拍照（USB + Orbbec深度相机）"
echo "时间: $TIMESTAMP"
echo "保存位置: $PHOTO_DIR"
echo "========================================="

# 1. 先用Orbbec SDK拍深度相机
echo ""
echo "--- 1. 拍摄Orbbec深度相机 ---"
python3 /home/agilex/code/yjw/cameras/side_camera.py

# 2. 再用fswebcam拍USB摄像头 (Dabai)
echo ""
echo "--- 2. 拍摄USB摄像头 (Dabai) ---"

CAMERA_INDEX=1
CAMERAS_FOUND=0
declare -A CAPTURED_BUS

for dev in /dev/video*; do
    if [ ! -c "$dev" ]; then
        continue
    fi
    
    dev_num=$(echo "$dev" | grep -o '[0-9]*$')
    dev_info=$(v4l2-ctl -d "$dev" --info 2>/dev/null)
    
    if ! echo "$dev_info" | grep -q "Driver name.*uvcvideo"; then
        continue
    fi
    
    card_type=$(echo "$dev_info" | grep "Card type" | sed 's/.*Card type\s*:\s*//' | xargs)
    bus_info=$(echo "$dev_info" | grep "Bus info" | sed 's/.*Bus info\s*:\s*//' | xargs)
    
    # 只处理Dabai USB摄像头
    if ! echo "$card_type" | grep -q "Dabai"; then
        continue
    fi
    
    # 去重
    if [ -n "${CAPTURED_BUS[$bus_info]}" ]; then
        continue
    fi
    
    CAPTURED_BUS[$bus_info]=$dev
    
    echo ""
    echo "USB摄像头 #$CAMERA_INDEX: $dev"
    echo "  类型: $card_type"
    echo "  端口: $bus_info"
    
    filename="usb_camera_${CAMERA_INDEX}_${dev_num}_${TIMESTAMP}.jpg"
    filepath="$PHOTO_DIR/$filename"
    
    echo "  拍照中..."
    if fswebcam -d "$dev" -r 1280x720 --no-banner "$filepath" 2>/dev/null; then
        echo "  ✓ 成功: $filename ($(du -h "$filepath" | cut -f1))"
        CAMERAS_FOUND=$((CAMERAS_FOUND + 1))
    else
        echo "  ✗ 失败"
    fi
    
    CAMERA_INDEX=$((CAMERA_INDEX + 1))
done

echo ""
echo "========================================="
echo "拍照完成！"
echo "  USB摄像头: $CAMERAS_FOUND 张"
echo "  Orbbec: 1 张 (用SDK)"
echo "照片保存在: $PHOTO_DIR"
echo "========================================="

echo ""
echo "所有照片:"
ls -lh "$PHOTO_DIR"/*_${TIMESTAMP}.jpg 2>/dev/null | nl