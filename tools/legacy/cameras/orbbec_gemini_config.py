import cv2
import numpy as np
import time
from datetime import datetime
import os
import json

class OrbbecGemini336L:
    """Orbbec Gemini 336L 相机控制器"""
    
    def __init__(self):
        # 根据测试结果配置
        self.color_dev = 12      # 彩色流
        self.depth_dev = 8       # 深度流（需要验证）
        self.ir_dev = 10         # IR流（需要验证）
        
        self.color_cap = None
        self.depth_cap = None
        self.ir_cap = None
        
        self.is_initialized = False
        
    def initialize(self):
        """初始化相机"""
        print("初始化 Orbbec Gemini 336L...")
        
        # 打开彩色相机
        self.color_cap = cv2.VideoCapture(self.color_dev)
        if not self.color_cap.isOpened():
            print(f"✗ 无法打开彩色流 /dev/video{self.color_dev}")
            return False
        print(f"✓ 彩色流已打开 /dev/video{self.color_dev}")
        
        # 打开深度相机（使用video8，需要验证）
        self.depth_cap = cv2.VideoCapture(self.depth_dev)
        if not self.depth_cap.isOpened():
            print(f"⚠️ 无法打开深度流 /dev/video{self.depth_dev}")
            # 继续，可能深度流在别的位置
        else:
            print(f"✓ 深度流已打开 /dev/video{self.depth_dev}")
        
        # 打开IR相机
        self.ir_cap = cv2.VideoCapture(self.ir_dev)
        if not self.ir_cap.isOpened():
            print(f"⚠️ 无法打开IR流 /dev/video{self.ir_dev}")
        else:
            print(f"✓ IR流已打开 /dev/video{self.ir_dev}")
        
        # 设置参数
        for cap in [self.color_cap, self.depth_cap, self.ir_cap]:
            if cap is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_initialized = True
        return True
    
    def capture_frame(self):
        """捕获一帧所有流"""
        if not self.is_initialized:
            return None
        
        result = {}
        
        # 捕获彩色
        if self.color_cap:
            ret, color = self.color_cap.read()
            if ret:
                result['color'] = color
        
        # 捕获深度
        if self.depth_cap:
            ret, depth_raw = self.depth_cap.read()
            if ret:
                # 处理深度数据
                # 将3通道转换为单通道深度
                if len(depth_raw.shape) == 3:
                    depth_gray = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
                else:
                    depth_gray = depth_raw
                result['depth'] = depth_gray
                result['depth_colored'] = self.colorize_depth(depth_gray)
        
        # 捕获IR
        if self.ir_cap:
            ret, ir = self.ir_cap.read()
            if ret:
                if len(ir.shape) == 3:
                    ir_gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
                else:
                    ir_gray = ir
                result['ir'] = ir_gray
        
        return result
    
    def colorize_depth(self, depth_data, max_depth=1000):
        """将深度数据彩色可视化"""
        # 归一化到0-255
        depth_normalized = cv2.normalize(
            depth_data.astype(np.float32), 
            None, 
            0, 255, 
            cv2.NORM_MINMAX
        )
        depth_uint8 = depth_normalized.astype(np.uint8)
        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        return depth_colored
    
    def capture_sequence(self, num_frames=30, save_dir=None):
        """捕获一系列帧"""
        if not self.is_initialized:
            print("相机未初始化")
            return
        
        if save_dir is None:
            save_dir = f"orbbec_sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n开始捕获 {num_frames} 帧...")
        print(f"保存到: {save_dir}")
        
        frames_collected = 0
        
        for i in range(num_frames):
            frame_data = self.capture_frame()
            
            if frame_data and 'color' in frame_data:
                timestamp = time.time()
                
                # 保存各流
                if 'color' in frame_data:
                    cv2.imwrite(
                        f"{save_dir}/color_{i:04d}.jpg", 
                        frame_data['color']
                    )
                
                if 'depth' in frame_data:
                    # 保存原始深度
                    np.save(
                        f"{save_dir}/depth_{i:04d}.npy", 
                        frame_data['depth']
                    )
                    # 保存彩色深度
                    cv2.imwrite(
                        f"{save_dir}/depth_colored_{i:04d}.jpg", 
                        frame_data['depth_colored']
                    )
                
                if 'ir' in frame_data:
                    cv2.imwrite(
                        f"{save_dir}/ir_{i:04d}.jpg", 
                        frame_data['ir']
                    )
                
                frames_collected += 1
                
                # 进度显示
                if (i + 1) % 10 == 0:
                    print(f"  已采集 {i + 1}/{num_frames} 帧")
            
            # 控制帧率
            time.sleep(0.033)
        
        print(f"\n✓ 完成! 共采集 {frames_collected} 帧")
        print(f"  保存目录: {save_dir}")
        
        # 保存元数据
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "num_frames": num_frames,
            "devices": {
                "color": self.color_dev,
                "depth": self.depth_dev,
                "ir": self.ir_dev
            }
        }
        with open(f"{save_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return save_dir
    
    def live_preview(self, duration=10):
        """实时预览"""
        if not self.is_initialized:
            print("相机未初始化")
            return
        
        print(f"\n开始实时预览 ({duration} 秒)...")
        print("按 'q' 或 'ESC' 退出")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                frame_data = self.capture_frame()
                
                if frame_data and 'color' in frame_data:
                    frame_count += 1
                    
                    # 显示彩色
                    cv2.imshow('Color', frame_data['color'])
                    
                    # 显示深度
                    if 'depth_colored' in frame_data:
                        cv2.imshow('Depth', frame_data['depth_colored'])
                    
                    # 显示IR
                    if 'ir' in frame_data:
                        cv2.imshow('IR', frame_data['ir'])
                    
                    # 显示帧率
                    if frame_count % 30 == 0:
                        print(f"  FPS: {frame_count / (time.time() - start_time):.1f}")
                
                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q 或 ESC
                    print("用户退出预览")
                    break
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("预览已中断")
        
        cv2.destroyAllWindows()
        print(f"\n预览结束，显示 {frame_count} 帧")
    
    def release(self):
        """释放所有资源"""
        for cap in [self.color_cap, self.depth_cap, self.ir_cap]:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()
        self.is_initialized = False
        print("所有相机已释放")

def main():
    # 创建相机实例
    camera = OrbbecGemini336L()
    
    # 初始化
    if not camera.initialize():
        print("初始化失败")
        return
    
    try:
        # 选项1: 实时预览
        print("\n选择模式:")
        print("  1: 实时预览 (10秒)")
        print("  2: 采集序列 (30帧)")
        print("  3: 两者都执行")
        
        choice = input("请选择 (1/2/3): ").strip()
        
        if choice == '1':
            camera.live_preview(duration=10)
        elif choice == '2':
            camera.capture_sequence(num_frames=30)
        elif choice == '3':
            camera.live_preview(duration=5)
            camera.capture_sequence(num_frames=30)
        else:
            print("无效选择，执行默认: 实时预览")
            camera.live_preview(duration=10)
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        camera.release()

if __name__ == "__main__":
    main()
