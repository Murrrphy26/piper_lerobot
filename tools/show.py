import cv2
import time
import os
import sys

def preview_to_file(dev_path='/dev/video12', width=640, height=480):
    """
    实时预览相机并保存到文件
    
    Args:
        dev_path: 相机设备路径
        width: 目标宽度
        height: 目标高度
    """
    cap = cv2.VideoCapture(dev_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开 {dev_path}")
        return
    
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 获取实际设置的分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ 相机已打开: {dev_path}")
    print(f"�� 目标分辨率: {width}x{height}")
    print(f"�� 实际分辨率: {actual_width}x{actual_height}")
    
    # 如果相机不支持目标分辨率，使用缩放
    use_resize = (actual_width != width or actual_height != height)
    if use_resize:
        print(f"⚠️ 相机不支持 {width}x{height}，将实时缩放到此分辨率")
    
    # 创建临时目录
    os.makedirs('camera_temp', exist_ok=True)
    
    # 统计信息
    frame_count = 0
    fps_count = 0
    fps_start = time.time()
    last_fps_print = time.time()
    
    print("\n按 Ctrl+C 停止预览")
    print("-" * 50)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n⚠️ 读取帧失败")
                break
            
            frame_count += 1
            fps_count += 1
            
            # 缩放到目标分辨率（如果需要）
            if use_resize:
                frame_resized = cv2.resize(frame, (width, height))
            else:
                frame_resized = frame
            
            # 保存当前帧
            cv2.imwrite('camera_temp/live.jpg', frame_resized)
            
            # 每秒打印一次 FPS
            current_time = time.time()
            if current_time - last_fps_print >= 1.0:
                fps = fps_count / (current_time - last_fps_print)
                h, w = frame.shape[:2]
                h_resized, w_resized = frame_resized.shape[:2]
                
                # 清空行并显示信息
                print(f"\r�� 原始: {w}x{h} -> 输出: {w_resized}x{h_resized} | "
                      f"FPS: {fps:.1f} | 总帧数: {frame_count} | "
                      f"时间: {time.strftime('%H:%M:%S')}", end='')
                
                # 重置计数
                fps_count = 0
                last_fps_print = current_time
            
            # 控制帧率（约 30 FPS，但保存文件会限制速度）
            time.sleep(0.001)  # 1ms 延迟
            
    except KeyboardInterrupt:
        print("\n\n�� 停止预览")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
    finally:
        cap.release()
        print(f"\n�� 总共处理了 {frame_count} 帧")
        print(f"�� 最新帧保存在: camera_temp/live.jpg")

if __name__ == "__main__":
    # 可以通过命令行参数指定相机设备
    dev = '/dev/video2\0'
    if len(sys.argv) > 1:
        dev = sys.argv[1]
    
    # 可以修改这里的分辨率为你需要的
    preview_to_file(dev, width=640, height=480)

