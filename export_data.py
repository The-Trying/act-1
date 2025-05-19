import os
import h5py
import cv2
import numpy as np
import pandas as pd

def export_hdf5_data(hdf5_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        # 提取控制命令（action）
        actions = f['/action'][()]
        num_timesteps = actions.shape[0]
        
        # 保存为CSV
        df_actions = pd.DataFrame(actions, columns=[f'command_{i}' for i in range(actions.shape[1])])
        df_actions.insert(0, 'timestep', np.arange(num_timesteps))  # 添加时间步列
        csv_actions_path = os.path.join(output_dir, 'actions.csv')
        df_actions.to_csv(csv_actions_path, index=False)
        print(f'Saved actions to: {csv_actions_path}')
        
        # 提取qpos数据并保存为CSV
        qpos = f['/observations/qpos'][()]
        df_qpos = pd.DataFrame(qpos, columns=[f'qpos_{i}' for i in range(qpos.shape[1])])
        df_qpos.insert(0, 'timestep', np.arange(qpos.shape[0]))  # 添加时间步列
        csv_qpos_path = os.path.join(output_dir, 'qpos.csv')
        df_qpos.to_csv(csv_qpos_path, index=False)
        print(f'Saved qpos to: {csv_qpos_path}')
        
        # 提取图像数据
        image_group = f['/observations/images']
        cam_names = list(image_group.keys())
        
        # 为每个摄像头创建子目录
        for cam_name in cam_names:
            cam_dir = os.path.join(output_dir, 'images', cam_name)
            os.makedirs(cam_dir, exist_ok=True)
            
            # 提取该摄像头的所有图像
            images = image_group[cam_name][()]  # (num_timesteps, H, W, C)
            
            # 按时间步保存图像
            for t in range(num_timesteps):
                image = images[t]
                # OpenCV需要BGR格式，如果原始数据是RGB需转换
                if image.shape[-1] == 3:  # 假设存储为RGB
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                filename = f"frame_{t:04d}.png"  # 时间步用4位数字补零
                cv2.imwrite(os.path.join(cam_dir, filename), image)
            print(f'Saved {len(images)} images to: {cam_dir}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_path', type=str, required=True, help='Input HDF5 file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    export_hdf5_data(args.hdf5_path, args.output_dir)