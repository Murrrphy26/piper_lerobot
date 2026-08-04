# tools/visualize_gripper_direct.py
"""
直接读取 parquet 文件，可视化夹爪角度变化
使用非交互式 matplotlib 后端，适合服务器环境
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
# 在导入 pyplot 之前设置后端
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
import json
import glob
import warnings
warnings.filterwarnings('ignore')

def load_parquet_data(dataset_path):
    """直接加载所有 parquet 文件"""
    data_dir = Path(dataset_path) / 'data' / 'chunk-000'
    parquet_files = sorted(data_dir.glob('*.parquet'))
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_data = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        all_data.append(df)
    
    if not all_data:
        return None
    
    # 合并所有数据
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total frames: {len(combined)}")
    
    return combined

def extract_gripper_data(df):
    """从 dataframe 提取夹爪数据"""
    # 从臂夹爪（observation.state）
    if 'observation.state' in df.columns:
        state_data = np.array(df['observation.state'].tolist())
        print(f"State data shape: {state_data.shape}")
        
        if state_data.shape[1] >= 14:
            follower_left = state_data[:, 6]
            follower_right = state_data[:, 13]
        else:
            follower_left = state_data[:, -2]
            follower_right = state_data[:, -1]
    else:
        print("Cannot find observation.state")
        return None
    
    # 主臂夹爪（action）
    if 'action' in df.columns:
        action_data = np.array(df['action'].tolist())
        print(f"Action data shape: {action_data.shape}")
        
        if action_data.shape[1] >= 14:
            leader_left = action_data[:, 6]
            leader_right = action_data[:, 13]
        else:
            leader_left = action_data[:, -2]
            leader_right = action_data[:, -1]
    else:
        print("Cannot find action")
        return None
    
    return {
        'follower_left': follower_left,
        'follower_right': follower_right,
        'leader_left': leader_left,
        'leader_right': leader_right,
        'diff_left': leader_left - follower_left,
        'diff_right': leader_right - follower_right,
        'steps': np.arange(len(follower_left))
    }

def plot_gripper_data(data_dict, title="Gripper Analysis", save_path=None):
    """绘制夹爪数据"""
    if data_dict is None:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    
    steps = data_dict['steps']
    
    # 图1：左夹爪
    ax1 = axes[0, 0]
    ax1.plot(steps, data_dict['leader_left'], label='Leader (主臂)', linewidth=2, alpha=0.8)
    ax1.plot(steps, data_dict['follower_left'], label='Follower (从臂)', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Gripper Position (m)')
    ax1.set_title('Left Gripper: Leader vs Follower')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2：右夹爪
    ax2 = axes[0, 1]
    ax2.plot(steps, data_dict['leader_right'], label='Leader (主臂)', linewidth=2, alpha=0.8)
    ax2.plot(steps, data_dict['follower_right'], label='Follower (从臂)', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Gripper Position (m)')
    ax2.set_title('Right Gripper: Leader vs Follower')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3：差异
    ax3 = axes[1, 0]
    ax3.plot(steps, data_dict['diff_left'], label='Left Diff', linewidth=2, color='red')
    ax3.plot(steps, data_dict['diff_right'], label='Right Diff', linewidth=2, color='blue')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Gripper Difference (m)')
    ax3.set_title('Leader - Follower Difference (Contact Indicator)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4：分布
    ax4 = axes[1, 1]
    diff_data = np.concatenate([data_dict['diff_left'], data_dict['diff_right']])
    ax4.hist(diff_data, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Zero difference')
    ax4.set_xlabel('Gripper Difference (m)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Gripper Differences')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved to: {save_path}")
    else:
        # 保存为默认文件名
        default_path = 'gripper_analysis.png'
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved to: {default_path}")
    
    plt.close(fig)  # 释放内存
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"Gripper Statistics")
    print(f"{'='*60}")
    print(f"Left Gripper:")
    print(f"  Leader (主臂) mean: {np.mean(data_dict['leader_left']):.4f} m")
    print(f"  Leader (主臂) std: {np.std(data_dict['leader_left']):.4f} m")
    print(f"  Follower (从臂) mean: {np.mean(data_dict['follower_left']):.4f} m")
    print(f"  Follower (从臂) std: {np.std(data_dict['follower_left']):.4f} m")
    print(f"  Difference mean: {np.mean(data_dict['diff_left']):.4f} m")
    print(f"  Difference max: {np.max(np.abs(data_dict['diff_left'])):.4f} m")
    
    print(f"\nRight Gripper:")
    print(f"  Leader (主臂) mean: {np.mean(data_dict['leader_right']):.4f} m")
    print(f"  Leader (主臂) std: {np.std(data_dict['leader_right']):.4f} m")
    print(f"  Follower (从臂) mean: {np.mean(data_dict['follower_right']):.4f} m")
    print(f"  Follower (从臂) std: {np.std(data_dict['follower_right']):.4f} m")
    print(f"  Difference mean: {np.mean(data_dict['diff_right']):.4f} m")
    print(f"  Difference max: {np.max(np.abs(data_dict['diff_right'])):.4f} m")
    
    # 接触检测（差异 > 2mm）
    contact_threshold = 0.002
    left_contact = np.where(np.abs(data_dict['diff_left']) > contact_threshold)[0]
    right_contact = np.where(np.abs(data_dict['diff_right']) > contact_threshold)[0]
    
    print(f"\n{'='*60}")
    print(f"Contact Detection (diff > {contact_threshold*1000:.0f}mm)")
    print(f"{'='*60}")
    print(f"Left gripper contact frames: {len(left_contact)} ({len(left_contact)/len(steps)*100:.1f}%)")
    print(f"Right gripper contact frames: {len(right_contact)} ({len(right_contact)/len(steps)*100:.1f}%)")
    
    if len(left_contact) > 0:
        print(f"  Left contact - first: {left_contact[0]}, last: {left_contact[-1]}")
    if len(right_contact) > 0:
        print(f"  Right contact - first: {right_contact[0]}, last: {right_contact[-1]}")
    
    # 判断是否有持续的夹爪接触（夹住物体）
    if len(left_contact) > len(steps) * 0.3:  # 超过30%的帧在接触
        print(f"\n⚠️  Left gripper appears to be CONSTANTLY IN CONTACT (夹住物体?)")
    if len(right_contact) > len(steps) * 0.3:
        print(f"\n⚠️  Right gripper appears to be CONSTANTLY IN CONTACT (夹住物体?)")
    
    return data_dict

def main():
    parser = argparse.ArgumentParser(description='Directly read parquet and visualize gripper data')
    parser.add_argument('--dataset-path', type=str, 
                        default='data/lerobot/local/cube_v4_3view_side',
                        help='Dataset directory path')
    parser.add_argument('--save-plot', type=str, default='gripper_analysis.png',
                        help='Save plot to file')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plot, only print statistics')
    
    args = parser.parse_args()
    
    # 加载数据
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        # 尝试从当前目录
        alt_path = Path.cwd() / args.dataset_path
        if alt_path.exists():
            dataset_path = alt_path
            print(f"✅ Found at: {dataset_path}")
        else:
            return
    
    print(f"Loading data from: {dataset_path}")
    df = load_parquet_data(dataset_path)
    
    if df is None or len(df) == 0:
        print("❌ No data loaded")
        return
    
    # 提取夹爪数据
    data = extract_gripper_data(df)
    
    if data is None:
        print("❌ Failed to extract gripper data")
        return
    
    # 打印数据范围
    print(f"\n{'='*60}")
    print(f"Data Range")
    print(f"{'='*60}")
    print(f"Left Gripper Range:")
    print(f"  Leader: [{np.min(data['leader_left']):.4f}, {np.max(data['leader_left']):.4f}]")
    print(f"  Follower: [{np.min(data['follower_left']):.4f}, {np.max(data['follower_left']):.4f}]")
    print(f"Right Gripper Range:")
    print(f"  Leader: [{np.min(data['leader_right']):.4f}, {np.max(data['leader_right']):.4f}]")
    print(f"  Follower: [{np.min(data['follower_right']):.4f}, {np.max(data['follower_right']):.4f}]")
    
    # 绘图
    if args.no_save:
        print("\nSkipping plot generation (--no-save)")
        # 仍然打印统计
        plot_gripper_data(data, title=f"Gripper Analysis - {dataset_path.name}", save_path=None)
    else:
        plot_gripper_data(data, title=f"Gripper Analysis - {dataset_path.name}", 
                         save_path=args.save_plot)
        print(f"\n✅ Analysis complete! Check the plot at: {args.save_plot}")

if __name__ == '__main__':
    main()