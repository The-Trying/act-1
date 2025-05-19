import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

# 读取图像的依赖（小怪兽）
import socket
import struct
import json
import threading
import cv2
import select

import IPython

e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 19
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        for ckpt_name in ckpt_names:
            eval_bc(config, ckpt_name, save_episode=True)
        print()
        exit()
    exit()

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def recvall(sock, n):
    """接收n字节数据"""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def receive_data(conn):
    """接收图像和位置信息"""
    # 接收图像长度
    img_len_bytes = recvall(conn, 4)
    if not img_len_bytes:
        return None, None
    img_len = struct.unpack('!i', img_len_bytes)[0]

    # 接收图像数据
    img_data = recvall(conn, img_len)
    if not img_data:
        return None, None

    # 接收meta长度
    meta_len_bytes = recvall(conn, 4)
    if not meta_len_bytes:
        return None, None
    meta_len = struct.unpack('!i', meta_len_bytes)[0]

    # 接收meta数据
    meta_data = recvall(conn, meta_len)
    if not meta_data:
        return None, None
    meta = json.loads(meta_data.decode())

    # 解码图像
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return frame, meta

def preprocess(frame, meta, stats):
    """预处理图像和位置信息"""
    # 图像预处理
    img = frame.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = torch.from_numpy(img).float().cuda().unsqueeze(0).unsqueeze(0)  # 添加batch维度

    # 位置信息预处理
    qpos = np.array([meta['bucketY'], meta['armY'], meta['boomY'], meta['cabY'], 
                     meta['coord1Position']['x'], meta['coord1Position']['y'], meta['coord1Position']['z'],
                     meta['coord2Position']['x'], meta['coord2Position']['y'], meta['coord2Position']['z'],
                     meta['coord3Position']['x'], meta['coord3Position']['y'], meta['coord3Position']['z'],
                     meta['coord4Position']['x'], meta['coord4Position']['y'], meta['coord4Position']['z'],
                     meta['coord5Position']['x'], meta['coord5Position']['y'], meta['coord5Position']['z']])
    qpos = (qpos - stats['qpos_mean']) / stats['qpos_std']
    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

    return img, qpos

def postprocess(action, stats):
    """后处理模型输出"""
    # action = action.detach().cpu().numpy().squeeze(0)
    action = action * stats['action_std'] + stats['action_mean']
    return action

def send_command(conn, action, meta):
    """发送控制命令到Unity"""
    """
    # action: 模型输出的“绝对”目标角度列表
    # meta: 最新的 meta，包含当前实际角度
    # """
    # 1) 取出当前角度
    curr = {
        'bucketY': meta['bucketY'],
        'armY':    meta['armY'],
        'boomY':   meta['boomY'],
        'cabY':    meta['cabY']
    }

    # 2) 计算最小角度变化
    cmd = {}
    for i, joint in enumerate(['bucketY','armY','boomY','cabY']):
        tgt = float(action[i])      # 模型给的“绝对”角度
        cur = float(curr[joint])    # Unity 传回的当前角度
        delta = (tgt - cur + 180) % 360 - 180
        if joint == 'cabY':
            # 特殊处理：cabY 的命令是固定速度
            if delta > 0.5:
                cmd[joint] = cur + 0.5
            elif delta < -0.5:
                cmd[joint] = cur - 0.5
            else:
                cmd[joint] = cur + delta
        else:
            # 其他关节按最短路径调整绝对角度
            cmd[joint] = cur + delta
        # 如果你的 Unity 接口接收“绝对”命令，就重新构造一个绝对角度：
        # cmd[joint] = cur + delta
        # 如果它接收“增量”命令，则直接 cmd[joint] = delta

    # 3) 序列化并发送
    data = json.dumps(cmd).encode('utf-8')
    hdr  = struct.pack('!i', len(data))
    conn.sendall(hdr + data)    
    print("cmd:")
    print(cmd)
    print("pos:")
    print(curr)
    print("tgt:")
    print(action)
    # """发送控制命令到Unity"""
    # cmd = {
    #     'bucketY': float(action[0]),
    #     'armY': float(action[1]),
    #     'boomY': float(action[2]),
    #     'cabY': float(action[3])
    # }
    # data = json.dumps(cmd).encode('utf-8')
    # hdr = struct.pack('!i', len(data))
    # conn.sendall(hdr + data)

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    max_timesteps = config['episode_len']
    temporal_agg = config['temporal_agg']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # # load environment
    # if real_robot:
    #     from aloha_scripts.robot_utils import move_grippers # requires aloha
    #     from aloha_scripts.real_env import make_real_env # requires aloha
    #     env = make_real_env(init_node=True)
    #     env_max_reward = 0
    # else:
    #     from sim_env import make_sim_env
    #     env = make_sim_env(task_name)
    #     env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    # 等待unity程序运行，并绑定连接
    host = '0.0.0.0'
    port = 6000

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[+] Listening on {host}:{port}")
    conn, addr = srv.accept()
    print(f"[+] Connected by {addr}")
    ### evaluation loop
    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

    qpos_list = []
    target_qpos_list = []
    with torch.inference_mode():
        for t in range(max_timesteps):
            frame, meta = receive_data(conn)
            if frame is None or meta is None:
                print("[warn] failed to receive, breaking")
                break
            img, qpos = preprocess(frame, meta, stats)
            curr_image = img
            print(curr_image.shape)
            ### query policy
            if config['policy_class'] == "ACT":
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = postprocess(raw_action, stats)
            target_qpos = action

            ### step the environment
            if t > 0:
                send_command(conn, action, meta) # 将控制命令发送到unity端
            ### for visualization
            qpos_list.append(qpos)
            target_qpos_list.append(target_qpos)
    # 断开连接，关闭通道
    conn.close()
    srv.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))
