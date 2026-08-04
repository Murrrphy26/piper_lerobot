#!/usr/bin/env python3
"""
使用Orbbec SDK拍摄深度相机照片
"""

import cv2
import numpy as np
import time
from datetime import datetime
import os

try:
    from pyorbbecsdk import *
    print("✓ Orbbec SDK 导入成功")
except ImportError:
    print("✗ 请安装 pyorbbecsdk: pip install pyorbbecsdk")
    exit(1)

def capture_orbbec_image():
    """使用Orbbec SDK捕获彩色图像"""
    
    # 创建照片目录
    photo_dir = os.path.expanduser("~/photos")
    os.makedirs(photo_dir, exist_ok=True)
    
    # 初始化SDK
    ctx = Context()
    if ctx is None:
        print("✗ 初始化Context失败")
        return None
    
    # 获取设备列表
    device_list = ctx.query_devices()
    if device_list is None or device_list.get_count() == 0:
        print("✗ 未找到Orbbec设备")
        return None
    
    print(f"✓ 找到 {device_list.get_count()} 个Orbbec设备")
    
    # 使用第一个设备
    device = device_list.get_device_by_index(0)
    if device is None:
        print("✗ 无法打开设备")
        return None
    
    print(f"✓ 设备: {device.get_device_info().get_name()}")
    
    # 配置颜色流
    color_profile_list = device.get_sensor_list().get_sensor_by_type(OBSensorType.COLOR_SENSOR).get_stream_profile_list()
    if color_profile_list is None:
        print("✗ 获取颜色流配置失败")
        return None
    
    # 选择最佳配置（1920x1080或1280x720）
    color_profile = color_profile_list.get_default_video_stream_profile()
    if color_profile is None:
        print("✗ 获取默认颜色流配置失败")
        return None
    
    # 配置流
    config = Config()
    config.enable_stream(OBStreamType.COLOR_STREAM, color_profile)
    
    # 启动管道
    pipeline = Pipeline(ctx)
    try:
        pipeline.start(config)
        print("✓ 管道启动成功")
    except Exception as e:
        print(f"✗ 启动管道失败: {e}")
        return None
    
    # 等待稳定
    time.sleep(1)
    
    # 捕获一帧
    print("正在捕获图像...")
    try:
        frames = pipeline.wait_for_frames(1000)  # 等待1秒
        if frames is None:
            print("✗ 未收到帧数据")
            pipeline.stop()
            return None
        
        color_frame = frames.get_color_frame()
        if color_frame is None:
            print("✗ 未收到彩色帧")
            pipeline.stop()
            return None
        
        # 转换为numpy数组
        color_data = np.asanyarray(color_frame.get_data())
        
        # 获取图像信息
        width = color_frame.get_width()
        height = color_frame.get_height()
        
        # 根据格式转换
        format = color_frame.get_format()
        if format == OBFormat.RGB888:
            img = color_data.reshape((height, width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif format == OBFormat.BGR888:
            img = color_data.reshape((height, width, 3))
        elif format == OBFormat.MJPG:
            # 如果是MJPG，需要解码
            img = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
        else:
            print(f"✗ 不支持的格式: {format}")
            pipeline.stop()
            return None
        
        pipeline.stop()
        return img
        
    except Exception as e:
        print(f"✗ 捕获失败: {e}")
        pipeline.stop()
        return None

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_dir = os.path.expanduser("photos")
    
    print("=========================================")
    print("Orbbec深度相机拍照")
    print(f"时间: {timestamp}")
    print("=========================================")
    
    # 捕获图像
    img = capture_orbbec_image()
    
    if img is not None:
        # 保存图像
        filename = f"orbbec_color_{timestamp}.jpg"
        filepath = os.path.join(photo_dir, filename)
        cv2.imwrite(filepath, img)
        
        print("")
        print("✓ 成功捕获Orbbec彩色图像")
        print(f"  文件: {filename}")
        print(f"  大小: {img.shape[1]}x{img.shape[0]}")
        print(f"  保存位置: {filepath}")
        
        # 显示图像信息
        file_size = os.path.getsize(filepath) / 1024
        print(f"  文件大小: {file_size:.1f} KB")
    else:
        print("✗ 捕获失败")

if __name__ == "__main__":
    main()