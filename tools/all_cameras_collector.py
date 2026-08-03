import cv2
import numpy as np
import time
from datetime import datetime
import os
import json
import threading
from queue import Queue

class AllCamerasCollector:
    """所有摄像头联合采集器（3个Piper + 1个Orbbec）"""
    
    def __init__(self, save_dir=None):
        if save_dir is None:
            save_dir = f"all_cameras_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Piper摄像头（来自之前的测试）
        self.piper_devices = [6, 8, 10]  # /dev/video0, 2, 4
        
        # Orbbec流
        self.orbbec_color = 12  # 彩色
        self.orbbec_depth = 8   # 深度（待验证）
        self.orbbec_ir = 10     # IR（待验证）
        
        self.caps = {}
        self.frame_queue = Queue()
        self.running = False
        
        print("=" * 60)
        print("所有摄像头采集器")
        print("=" * 60)
        print(f"Piper摄像头: {self.piper_devices}")
        print(f"Orbbec彩色: /dev/video{self.orbbec_color}")
        print(f"Orbbec深度: /dev/video{self.orbbec_depth}")
        print(f"Orbbec IR:   /dev/video{self.orbbec_ir}")
        print(f"保存目录: {self.save_dir}")
        print("=" * 60)
        
    def open_cameras(self):
        """打开所有相机"""
        # 打开Piper摄像头
        for dev_id in self.piper_devices:
            cap = cv2.VideoCapture(dev_id)
            if cap.isOpened():
                self.caps[f'piper_{dev_id}'] = cap
                print(f"✓ Piper /dev/video{dev_id} 已打开")
            else:
                print(f"✗ Piper /dev/video{dev_id} 打开失败")
        
        # 打开Orbbec彩色
        cap = cv2.VideoCapture(self.orbbec_color)
        if cap.isOpened():
            self.caps['orbbec_color'] = cap
            print(f"✓ Orbbec彩色 /dev/video{self.orbbec_color} 已打开")
        else:
            print(f"✗ Orbbec彩色 /dev/video{self.orbbec_color} 打开失败")
        
        # 打开Orbbec深度
        cap = cv2.VideoCapture(self.orbbec_depth)
        if cap.isOpened():
            self.caps['orbbec_depth'] = cap
            print(f"✓ Orbbec深度 /dev/video{self.orbbec_depth} 已打开")
        else:
            print(f"⚠️ Orbbec深度 /dev/video{self.orbbec_depth} 打开失败")
        
        # 打开Orbbec IR
        cap = cv2.VideoCapture(self.orbbec_ir)
        if cap.isOpened():
            self.caps['orbbec_ir'] = cap
            print(f"✓ Orbbec IR /dev/video{self.orbbec_ir} 已打开")
        else:
            print(f"⚠️ Orbbec IR /dev/video{self.orbbec_ir} 打开失败")
        
        return len(self.caps) > 0
    
    def capture_sequence(self, num_frames=30):
        """捕获图像序列"""
        if not self.caps:
            if not self.open_cameras():
                print("没有相机可用")
                return
        
        print(f"\n开始采集 {num_frames} 帧...")
        print("按 Ctrl+C 停止\n")
        
        frame_counts = {name: 0 for name in self.caps.keys()}
        
        try:
            for i in range(num_frames):
                timestamp = time.time()
                frame_data = {}
                
                # 从每个相机捕获
                for name, cap in self.caps.items():
                    ret, frame = cap.read()
                    if ret:
                        frame_data[name] = frame
                        frame_counts[name] += 1
                
                # 保存所有帧
                if frame_data:
                    timestamp_str = f"{timestamp:.6f}"
                    
                    for name, frame in frame_data.items():
                        filename = f"{self.save_dir}/{name}_{timestamp_str}.jpg"
                        cv2.imwrite(filename, frame)
                    
                    # 进度
                    if (i + 1) % 10 == 0:
                        print(f"  已采集 {i + 1}/{num_frames} 帧")
                else:
                    print(f"⚠️ 第 {i+1} 帧采集失败")
                
                # 控制帧率
                time.sleep(0.033)
                
        except KeyboardInterrupt:
            print("\n用户中断采集")
        
        print("\n" + "=" * 60)
        print("采集完成！统计信息:")
        for name, count in frame_counts.items():
            print(f"  {name}: {count} 帧")
        print(f"总帧数: {sum(frame_counts.values())}")
        print(f"数据保存到: {self.save_dir}")
        print("=" * 60)
        
        # 保存元数据
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "num_frames": num_frames,
            "frame_counts": frame_counts,
            "devices": list(self.caps.keys())
        }
        with open(f"{self.save_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def release(self):
        """释放所有相机"""
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()
        print("所有相机已释放")

def main():
    # 检查保存目录
    save_dir = input("保存目录名 (直接回车使用自动生成): ").strip()
    if not save_dir:
        save_dir = None
    
    collector = AllCamerasCollector(save_dir)
    
    try:
        # 询问采集帧数
        num_frames = input("采集帧数 (默认30): ").strip()
        if not num_frames:
            num_frames = 30
        else:
            num_frames = int(num_frames)
        
        collector.capture_sequence(num_frames)
        
    except KeyboardInterrupt:
        print("\n程序被中断")
    finally:
        collector.release()

if __name__ == "__main__":
    main()
