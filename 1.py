from __future__ import annotations

import os
import pdb
import random
import time
from typing import Literal
from dataclasses import dataclass, asdict, make_dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import pickle
import json
import copy
import trimesh

from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import WeightedPrimitiveSequenceDataset, SinglePrimitiveDataset
from utils.smpl_utils import *
from utils.misc_util import encode_text, compose_texts_with_and
from pytorch3d import transforms
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.resample import create_named_schedule_sampler

from mld.train_mvae import Args as MVAEArgs
from mld.train_mvae import DataArgs, TrainArgs
from mld.train_mld import DenoiserArgs, MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs
from mld.rollout_mld import load_mld, ClassifierFreeWrapper

import smplx
from VolumetricSMPL import attach_volume
from pytorch3d.transforms import matrix_to_axis_angle

debug = 0

# 增加场景采样密度
NUM_POINTS_SAMPLE_FOR_VOLSMPL = 3000  # 从100增加到3000

# 添加新的配置参数
COLLISION_DETECTION_RADIUS = 2.5  # 增加检测范围到2.5米
FRAME_SKIP_INTERVAL = 2  # 每两帧检测一次

# SMPLX关节名称映射（54个关节）
SMPLX_JOINT_NAMES = {
    0: 'pelvis', 1: 'left_hip', 2: 'right_hip', 3: 'spine1', 
    4: 'left_knee', 5: 'right_knee', 6: 'spine2', 7: 'left_ankle',
    8: 'right_ankle', 9: 'spine3', 10: 'left_foot', 11: 'right_foot',
    12: 'neck', 13: 'left_collar', 14: 'right_collar', 15: 'head',
    16: 'left_shoulder', 17: 'right_shoulder', 18: 'left_elbow', 19: 'right_elbow',
    20: 'left_wrist', 21: 'right_wrist', 22: 'jaw', 23: 'left_eye_smplhf', 
    24: 'right_eye_smplhf',
    # 手指关节
    25: 'left_index1', 26: 'left_index2', 27: 'left_index3',
    28: 'left_middle1', 29: 'left_middle2', 30: 'left_middle3',
    31: 'left_pinky1', 32: 'left_pinky2', 33: 'left_pinky3',
    34: 'left_ring1', 35: 'left_ring2', 36: 'left_ring3',
    37: 'left_thumb1', 38: 'left_thumb2', 39: 'left_thumb3',
    40: 'right_index1', 41: 'right_index2', 42: 'right_index3',
    43: 'right_middle1', 44: 'right_middle2', 45: 'right_middle3',
    46: 'right_pinky1', 47: 'right_pinky2', 48: 'right_pinky3',
    49: 'right_ring1', 50: 'right_ring2', 51: 'right_ring3',
    52: 'right_thumb1', 53: 'right_thumb2', 54: 'right_thumb3'
}

# 关节到身体部位的映射
JOINT_TO_BODY_PART = {
    # 头部和面部
    15: 'head', 22: 'head', 23: 'head', 24: 'head',
    # 颈部
    12: 'neck',
    # 躯干
    0: 'pelvis', 3: 'lower_torso', 6: 'upper_torso', 9: 'chest',
    # 左臂
    13: 'left_shoulder', 16: 'left_shoulder', 18: 'left_upper_arm', 
    20: 'left_forearm',
    # 右臂
    14: 'right_shoulder', 17: 'right_shoulder', 19: 'right_upper_arm',
    21: 'right_forearm',
    # 左腿
    1: 'left_hip', 4: 'left_thigh', 7: 'left_shin', 10: 'left_foot',
    # 右腿
    2: 'right_hip', 5: 'right_thigh', 8: 'right_shin', 11: 'right_foot',
    # 左手
    25: 'left_hand', 26: 'left_hand', 27: 'left_hand', 28: 'left_hand',
    29: 'left_hand', 30: 'left_hand', 31: 'left_hand', 32: 'left_hand',
    33: 'left_hand', 34: 'left_hand', 35: 'left_hand', 36: 'left_hand',
    37: 'left_hand', 38: 'left_hand', 39: 'left_hand',
    # 右手
    40: 'right_hand', 41: 'right_hand', 42: 'right_hand', 43: 'right_hand',
    44: 'right_hand', 45: 'right_hand', 46: 'right_hand', 47: 'right_hand',
    48: 'right_hand', 49: 'right_hand', 50: 'right_hand', 51: 'right_hand',
    52: 'right_hand', 53: 'right_hand', 54: 'right_hand'
}

def get_body_part_from_vertex(vertex_pos, joints_pos, vertices_tensor):
    """基于顶点位置和关节位置推断身体部位"""
    # 计算顶点到各个关节的距离
    distances = torch.cdist(vertex_pos.unsqueeze(0), joints_pos.unsqueeze(0))[0]
    
    # 找到最近的关节
    nearest_joint_idx = torch.argmin(distances).item()
    
    # 映射到身体部位
    if nearest_joint_idx in JOINT_TO_BODY_PART:
        return JOINT_TO_BODY_PART[nearest_joint_idx]
    else:
        # 对于未映射的关节，返回通用部位
        if nearest_joint_idx < 25:
            return 'body'
        elif nearest_joint_idx < 40:
            return 'left_hand'
        else:
            return 'right_hand'

def get_collision_body_parts_v2(collision_mask, vertices, joints):
    """基于关节距离的方法来推断碰撞的身体部位"""
    collision_indices = torch.where(collision_mask)[0]
    body_parts = {}
    
    if len(collision_indices) == 0:
        return body_parts
    
    # 获取碰撞顶点的位置
    collision_vertices = vertices[collision_indices]
    
    for i, vertex_idx in enumerate(collision_indices):
        vertex_pos = collision_vertices[i]
        part = get_body_part_from_vertex(vertex_pos, joints[0], vertices)
        
        if part not in body_parts:
            body_parts[part] = 0
        body_parts[part] += 1
    
    return body_parts

@dataclass
class OptimArgs:
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir = None

    denoiser_checkpoint: str = ''

    respacing: str = 'ddim10'
    guidance_param: float = 5.0
    export_smpl: int = 0
    zero_noise: int = 0
    use_predicted_joints: int = 0
    batch_size: int = 1

    optim_lr: float = 0.01
    optim_steps: int = 100 # 300
    optim_unit_grad: int = 1
    optim_anneal_lr: int = 1
    weight_jerk: float = 0.0
    weight_collision: float = 0.5  # 默认使用较高的碰撞权重
    weight_contact: float = 0.0
    weight_skate: float = 0.0
    load_cache: int = 0
    contact_thresh: float = 0.03
    init_noise_scale: float = 1.0

    interaction_cfg: str = './data/optim_interaction/climb_up_stairs.json'


import torch.nn.functional as F

def batchify_smpl_output(smpl_output):
    b_smpl_output_list = []
    batch_size = smpl_output.vertices.shape[0]
    for b_ind in range(batch_size):
        b_smpl_output_list.append(copy.copy(smpl_output))
        for key in b_smpl_output_list[-1].keys():
            val = getattr(smpl_output, key)
            if torch.is_tensor(val):
                val = val[b_ind:b_ind+1].clone()
            setattr(b_smpl_output_list[-1], key, val)
    return b_smpl_output_list

def filter_scene_points_around_full_body(scene_points, joints, radius=COLLISION_DETECTION_RADIUS):
    """改进的场景点过滤：基于所有关节位置而非只用pelvis"""
    device = scene_points.device
    scene_pts = scene_points.squeeze(0)  # [N, 3]
    
    if joints.dim() == 3:  # [B, J, 3]
        all_joints = joints[0]  # [J, 3] 取第一个batch的关节
    else:  # [J, 3]
        all_joints = joints
    
    # 计算场景点到所有关节的最小距离
    distances = torch.cdist(scene_pts.unsqueeze(0), all_joints.unsqueeze(0))[0]  # [N, J]
    min_distances = distances.min(dim=1)[0]  # [N]
    
    # 保留距离任意关节小于radius的点
    mask = min_distances <= radius
    
    # 确保至少有足够的点进行碰撞检测
    if mask.sum() < 500:
        _, indices = torch.topk(min_distances, k=min(2000, len(min_distances)), largest=False)
        mask = torch.zeros_like(min_distances, dtype=torch.bool)
        mask[indices] = True
    
    filtered_points = scene_points[:, mask, :]
    return filtered_points

def filter_scene_points_around_body(scene_points, body_center, radius=COLLISION_DETECTION_RADIUS):
    """原始的基于单点的过滤方法（保持兼容性）"""
    distances = torch.norm(scene_points.squeeze(0) - body_center.detach(), dim=1)
    mask = distances <= radius
    indices = torch.where(mask)[0]
    
    if indices.numel() == 0:
        # 如果没有点在范围内，返回最近的1000个点
        _, nearest_indices = torch.topk(distances, k=min(1000, len(distances)), largest=False)
        indices = nearest_indices
    
    filtered_points = scene_points[:, indices, :]
    return filtered_points

def calc_point_sdf(scene_assets, points):
    device = points.device
    scene_sdf_config = scene_assets['scene_sdf_config']
    scene_sdf_grid = scene_assets['scene_sdf_grid']
    sdf_size = scene_sdf_config['size']
    sdf_scale = scene_sdf_config['scale']
    sdf_scale = torch.tensor(sdf_scale, dtype=torch.float32, device=device).reshape(1, 1, 1)  # [1, 1, 1]
    sdf_center = scene_sdf_config['center']
    sdf_center = torch.tensor(sdf_center, dtype=torch.float32, device=device).reshape(1, 1, 3)  # [1, 1, 3]
    batch_size, num_points, _ = points.shape
    # convert to [-1, 1], here scale is (1.6/extent) proportional to the inverse of scene size, https://github.com/wang-ps/mesh2sdf/blob/1b54d1f5458d8622c444f78d4477f600a6fe50e1/example/test.py#L22
    points = (points - sdf_center) * sdf_scale  # [B, num_points, 3]
    sdf_values = F.grid_sample(scene_sdf_grid.unsqueeze(0),  # [B, 1, size, size, size]
                               points[:, :, [2, 1, 0]].view(batch_size, num_points, 1, 1, 3),
                               padding_mode='border',
                               align_corners=True
                               ).reshape(batch_size, num_points)
    # print('sdf_values', sdf_values.shape)
    sdf_values = sdf_values / sdf_scale.squeeze(-1)  # [B, P], scale back to the original scene size
    return sdf_values

def calc_jerk(joints):
    vel = joints[:, 1:] - joints[:, :-1]  # --> B x T-1 x 22 x 3
    acc = vel[:, 1:] - vel[:, :-1]  # --> B x T-2 x 22 x 3
    jerk = acc[:, 1:] - acc[:, :-1]  # --> B x T-3 x 22 x 3
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))  # --> B x T-3 x 22, compute L1 norm of jerk
    jerk = jerk.amax(dim=[1, 2])  # --> B, Get the max of the jerk across all joints and frames

    return jerk.mean()

def optimize(history_motion_tensor, transf_rotmat, transf_transl, text_prompt, goal_joints, joints_mask):
    texts = []
    if ',' in text_prompt:  # contain a time line of multipel actions
        num_rollout = 0
        for segment in text_prompt.split(','):
            action, num_mp = segment.split('*')
            action = compose_texts_with_and(action.split(' and '))
            texts = texts + [action] * int(num_mp)
            num_rollout += int(num_mp)
    else:
        action, num_rollout = text_prompt.split('*')
        action = compose_texts_with_and(action.split(' and '))
        num_rollout = int(num_rollout)
        for _ in range(num_rollout):
            texts.append(action)
    all_text_embedding = encode_text(dataset.clip_model, texts, force_empty_zero=True).to(dtype=torch.float32,
                                                                                      device=device)

    def rollout(noise, history_motion_tensor, transf_rotmat, transf_transl):
        motion_sequences = None
        history_motion = history_motion_tensor
        for segment_id in range(num_rollout):
            text_embedding = all_text_embedding[segment_id].expand(batch_size, -1)  # [B, 512]
            guidance_param = torch.ones(batch_size, *denoiser_args.model_args.noise_shape).to(device=device) * optim_args.guidance_param
            y = {
                'text_embedding': text_embedding,
                'history_motion_normalized': history_motion,
                'scale': guidance_param,
            }

            x_start_pred = sample_fn(
                denoiser_model,
                (batch_size, *denoiser_args.model_args.noise_shape),
                clip_denoised=False,
                model_kwargs={'y': y},
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                noise=noise[segment_id],
            )  # [B, T=1, D]
            # x_start_pred = x_start_pred.clamp(min=-3, max=3)
            # print('x_start_pred:', x_start_pred.mean(), x_start_pred.std(), x_start_pred.min(), x_start_pred.max())
            latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
            future_motion_pred = vae_model.decode(latent_pred, history_motion, nfuture=future_length,
                                                       scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized

            future_frames = dataset.denormalize(future_motion_pred)
            new_history_frames = future_frames[:, -history_length:, :]

            """transform primitive to world coordinate, prepare for serialization"""
            if segment_id == 0:  # add init history motion
                future_frames = torch.cat([dataset.denormalize(history_motion), future_frames], dim=1)
            future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
            future_feature_dict.update(
                {
                    'transf_rotmat': transf_rotmat,
                    'transf_transl': transf_transl,
                    'gender': gender,
                    'betas': betas[:, :future_length, :] if segment_id > 0 else betas[:, :primitive_length, :],
                    'pelvis_delta': pelvis_delta,
                }
            )
            future_primitive_dict = primitive_utility.feature_dict_to_smpl_dict(future_feature_dict)
            future_primitive_dict = primitive_utility.transform_primitive_to_world(future_primitive_dict)
            if motion_sequences is None:
                motion_sequences = future_primitive_dict
            else:
                for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
                    motion_sequences[key] = torch.cat([motion_sequences[key], future_primitive_dict[key]], dim=1)  # [B, T, ...]

            """update history motion seed, update global transform"""
            history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames)
            history_feature_dict.update(
                {
                    'transf_rotmat': transf_rotmat,
                    'transf_transl': transf_transl,
                    'gender': gender,
                    'betas': betas[:, :history_length, :],
                    'pelvis_delta': pelvis_delta,
                }
            )
            canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
                history_feature_dict, use_predicted_joints=optim_args.use_predicted_joints)
            transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
            canonicalized_history_primitive_dict['transf_transl']
            history_motion = primitive_utility.dict_to_tensor(blended_feature_dict)
            history_motion = dataset.normalize(history_motion)  # [B, T, D]

        return motion_sequences, history_motion, transf_rotmat, transf_transl

    optim_steps = optim_args.optim_steps
    lr = optim_args.optim_lr
    noise = torch.randn(num_rollout, batch_size, *denoiser_args.model_args.noise_shape,
                        device=device, dtype=torch.float32)
    # noise = noise.clip(min=-1, max=1)
    noise = noise * optim_args.init_noise_scale
    noise.requires_grad_(True)
    reduction_dims = list(range(1, len(noise.shape)))
    criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)

    optimizer = torch.optim.Adam([noise], lr=lr)
    for i in tqdm(range(optim_steps)):
        optimizer.zero_grad()
        if optim_args.optim_anneal_lr:
            frac = 1.0 - i / optim_steps
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow

        motion_sequences, new_history_motion_tensor, new_transf_rotmat, new_transf_transl = rollout(noise,
                                                                                                    history_motion_tensor,
                                                                                                    transf_rotmat,
                                                                                                    transf_transl)
        global_joints = motion_sequences['joints']  # [B, T, 22, 3]
        B, T, _, _ = global_joints.shape

        _, _, J, _, _ = motion_sequences["body_pose"].shape

        # 只在选定的帧上进行碰撞检测（每FRAME_SKIP_INTERVAL帧）
        frame_indices = list(range(0, T, FRAME_SKIP_INTERVAL))
        print(f"\n  [DEBUG] 步骤 {i+1}/{optim_steps}: 检测帧 {frame_indices} (共{len(frame_indices)}帧)")
        
        loss_collision = 0
        collision_count = 0
        total_collision_checks = 0
        
        # 创建/获取SMPL模型（只创建一次）
        if 'model' not in scene_assets:
            model = smplx.create(model_path="./data/smplx_lockedhead_20230207/models_lockedhead/smplx/SMPLX_NEUTRAL.npz", 
                               gender='neutral', use_pca=True, num_pca_comps=12, num_betas=10, batch_size=1).to(device)
            scene_assets['model'] = attach_volume(model, pretrained=True, device=device)
        
        for frame_idx in frame_indices:
            if frame_idx >= T:
                continue
            
            frame_collision_loss = 0
            frame_collision_count = 0
            
            # 获取当前帧的关节位置用于改进的场景点过滤
            current_joints = motion_sequences['joints'][:, frame_idx, :, :]  # [B, 22, 3]
            
            print(f"    [DEBUG] 帧 {frame_idx}: 处理 {B} 个batch")
            
            # 为每个batch单独处理（修复的关键部分）
            for batch_idx in range(B):
                total_collision_checks += 1
                
                # 获取单个batch的参数
                batch_transl = motion_sequences["transl"][batch_idx:batch_idx+1, frame_idx, :].reshape(1, 3)
                batch_global_orient = matrix_to_axis_angle(
                    motion_sequences["global_orient"][batch_idx:batch_idx+1, frame_idx, :, :]).reshape(1, 3)
                batch_body_pose = matrix_to_axis_angle(
                    motion_sequences["body_pose"][batch_idx:batch_idx+1, frame_idx, :, :, :]).reshape(1, J*3)
                
                # 基于该batch的关节位置过滤场景点
                batch_joints = current_joints[batch_idx:batch_idx+1, :, :]  # [1, 22, 3]
                filtered_scene_points = filter_scene_points_around_full_body(
                    scene_assets['sampled_points'], 
                    batch_joints, 
                    radius=COLLISION_DETECTION_RADIUS
                )
                
                # 运行SMPL前向传播（单个batch）
                smpl_output = scene_assets['model'](
                    transl=batch_transl,
                    global_orient=batch_global_orient,
                    body_pose=batch_body_pose,
                    return_verts=True, return_full_pose=True)
                
                # 检测碰撞
                loss, collision_mask = scene_assets['model'].volume.collision_loss(
                    filtered_scene_points, smpl_output, ret_collision_mask=True)
                
                if loss > 0:
                    frame_collision_loss += loss
                    frame_collision_count += 1
                    collision_count += 1
                    
                    # 详细的碰撞信息
                    if collision_mask is not None:
                        collision_parts = get_collision_body_parts_v2(
                            collision_mask, 
                            smpl_output.vertices[0],  # 当前batch的顶点
                            smpl_output.joints        # 当前batch的关节
                        )
                        if collision_parts:
                            print(f"      [COLLISION] Batch {batch_idx}: 损失 {loss.item():.6f}")
                            print(f"        身体部位: {', '.join([f'{part}({count}个顶点)' for part, count in collision_parts.items()])}")
                            print(f"        Batch位置: {batch_transl[0].detach().cpu().numpy()}")
                            print(f"        场景点数: {filtered_scene_points.shape[1]}")
                
                # 清理内存
                del loss
                del smpl_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            loss_collision += frame_collision_loss

        # 平均碰撞损失
        if len(frame_indices) > 0:
            loss_collision = loss_collision / len(frame_indices)
        
        if loss_collision == 0.0:
            loss_collision = torch.tensor(loss_collision, device=device)

        joints_sdf = calc_point_sdf(scene_assets, global_joints.reshape(1, -1, 3)).reshape(B, T, 22)
        foot_joints_sdf = joints_sdf[:, :, FOOT_JOINTS_IDX]  # [B, T, 2]
        loss_floor_contact = (foot_joints_sdf.amin(dim=-1) - optim_args.contact_thresh).clamp(min=0).mean()

        loss_joints = criterion(motion_sequences['joints'][:, -1, joints_mask], goal_joints[:, joints_mask])
        loss_jerk = calc_jerk(motion_sequences['joints'])
        loss = loss_joints + optim_args.weight_collision * loss_collision + optim_args.weight_jerk * loss_jerk + optim_args.weight_contact * loss_floor_contact

        loss.backward()
        if optim_args.optim_unit_grad:
            noise.grad.data /= noise.grad.norm(p=2, dim=reduction_dims, keepdim=True).clamp(min=1e-6)
        optimizer.step()
        
        print(f'  [SUMMARY] 步骤 {i+1}/{optim_steps}:')
        print(f'    总损失: {loss.item():.6f}')
        print(f'    目标损失: {loss_joints.item():.6f}')
        print(f'    碰撞损失: {loss_collision.item() if hasattr(loss_collision, "item") else loss_collision:.6f} (权重: {optim_args.weight_collision})')
        print(f'    碰撞检测: {collision_count}/{total_collision_checks} (检测到碰撞/总检测次数)')
        print(f'    运动平滑度: {loss_jerk.item():.6f} (权重: {optim_args.weight_jerk})')
        print(f'    地面接触: {loss_floor_contact.item():.6f} (权重: {optim_args.weight_contact})')

    for key in motion_sequences:
        if torch.is_tensor(motion_sequences[key]):
            motion_sequences[key] = motion_sequences[key].detach()
    motion_sequences['texts'] = texts

    return motion_sequences, new_history_motion_tensor.detach(), new_transf_rotmat.detach(), new_transf_transl.detach()

if __name__ == '__main__':

    optim_args = tyro.cli(OptimArgs)
    # TRY NOT TO MODIFY: seeding
    random.seed(optim_args.seed)
    np.random.seed(optim_args.seed)
    torch.manual_seed(optim_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = optim_args.torch_deterministic
    device = torch.device(optim_args.device if torch.cuda.is_available() else "cpu")
    optim_args.device = device

    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(optim_args.denoiser_checkpoint, device)
    denoiser_checkpoint = Path(optim_args.denoiser_checkpoint)
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / 'optim'
    save_dir.mkdir(parents=True, exist_ok=True)
    optim_args.save_dir = save_dir

    diffusion_args = denoiser_args.diffusion_args
    assert 'ddim' in optim_args.respacing
    diffusion_args.respacing = optim_args.respacing
    print('diffusion_args:', asdict(diffusion_args))
    diffusion = create_gaussian_diffusion(diffusion_args)
    sample_fn = diffusion.ddim_sample_loop_full_chain

    # load initial seed dataset
    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,  # cfg path from model checkpoint
                                     dataset_path=vae_args.data_args.data_dir,  # dataset path from model checkpoint
                                     sequence_path='./data/stand.pkl',
                                     batch_size=optim_args.batch_size,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     )
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    primitive_utility = dataset.primitive_utility
    batch_size = optim_args.batch_size

    with open('./data/joint_skin_dist.json', 'r') as f:
        joint_skin_dist = json.load(f)
        joint_skin_dist = torch.tensor(joint_skin_dist, dtype=torch.float32, device=device)
        joint_skin_dist = joint_skin_dist.clamp(min=optim_args.contact_thresh)  # [22]

    """optimization config"""
    with open(optim_args.interaction_cfg, 'r') as f:
        interaction_cfg = json.load(f)
    interaction_name = interaction_cfg['interaction_name'].replace(' ', '_')
    scene_dir = Path(interaction_cfg['scene_dir'])
    scene_dir = Path(scene_dir)
    scene_with_floor_mesh = trimesh.load(scene_dir / 'scene_with_floor.obj', process=False, force='mesh')

    with open(scene_dir / 'scene_sdf.json', 'r') as f:
        scene_sdf_config = json.load(f)
    scene_sdf_grid = np.load(scene_dir / 'scene_sdf.npy')
    scene_sdf_grid = torch.tensor(scene_sdf_grid, dtype=torch.float32, device=device).unsqueeze(
        0)  # [1, size, size, size]

    sampled_points = scene_with_floor_mesh.vertices
    sampled_points = torch.from_numpy(sampled_points).float().to(device).reshape(1, -1, 3)
    sampled_points_2 = scene_with_floor_mesh.sample(NUM_POINTS_SAMPLE_FOR_VOLSMPL)
    sampled_points_2 = torch.from_numpy(sampled_points_2).float().to(device).reshape(1, -1, 3)

    sampled_points = torch.cat([sampled_points, sampled_points_2], dim=1)

    print(f"[INIT] 改进的碰撞检测配置:")
    print(f"  检测半径: {COLLISION_DETECTION_RADIUS}m")
    print(f"  帧检测间隔: {FRAME_SKIP_INTERVAL}")
    print(f"  总场景点数: {sampled_points.shape[1]}")
    print(f"  额外采样点数: {NUM_POINTS_SAMPLE_FOR_VOLSMPL}")
    print(f"  使用改进的全身关节过滤方法")

    scene_assets = {
        'sampled_points': sampled_points,
        'scene_with_floor_mesh': scene_with_floor_mesh,
        'scene_sdf_grid': scene_sdf_grid,
        'scene_sdf_config': scene_sdf_config,
        'floor_height': interaction_cfg['floor_height'],
    }

    out_path = optim_args.save_dir
    filename = f'guidance{optim_args.guidance_param}_seed{optim_args.seed}'
    if optim_args.respacing != '':
        filename = f'{optim_args.respacing}_{filename}'
    if optim_args.zero_noise:
        filename = f'zero_noise_{filename}'
    if optim_args.use_predicted_joints:
        filename = f'use_pred_joints_{filename}'
    filename = f'{interaction_name}_{filename}'
    filename = f'{filename}_contact{optim_args.weight_contact}_thresh{optim_args.contact_thresh}_collision{optim_args.weight_collision}_jerk{optim_args.weight_jerk}'
    out_path = out_path / filename
    out_path.mkdir(parents=True, exist_ok=True)

    batch = dataset.get_batch(batch_size=optim_args.batch_size)
    input_motions, model_kwargs = batch[0]['motion_tensor_normalized'], {'y': batch[0]}
    del model_kwargs['y']['motion_tensor_normalized']
    gender = model_kwargs['y']['gender'][0]
    betas = model_kwargs['y']['betas'][:, :primitive_length, :].to(device)  # [B, H+F, 10]
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': betas[:, 0, :],
        'gender': gender,
    })
    input_motions = input_motions.to(device)  # [B, D, 1, T]
    motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)  # [B, T, D]
    init_history_motion = motion_tensor[:, :history_length, :]  # [B, H, D]

    all_motion_sequences = None
    for interaction_idx, interaction in enumerate(interaction_cfg['interactions']):
        cache_path = out_path / f'cache_{interaction_idx}.pkl'
        if cache_path.exists() and optim_args.load_cache:
            with open(cache_path, 'rb') as f:
                all_motion_sequences, history_motion_tensor, transf_rotmat, transf_transl = pickle.load(f)
            tensor_dict_to_device(all_motion_sequences, device)
            history_motion_tensor = history_motion_tensor.to(device)
            transf_rotmat = transf_rotmat.to(device)
            transf_transl = transf_transl.to(device)
        else:
            text_prompt = interaction['text_prompt']
            goal_joints = torch.zeros(batch_size, 22, 3, device=device, dtype=torch.float32)
            goal_joints[:, 0] = torch.tensor(interaction['goal_joints'][0], device=device, dtype=torch.float32)
            joints_mask = torch.zeros(22, device=device, dtype=torch.bool)
            joints_mask[0] = 1

            if interaction_idx == 0:
                history_motion_tensor = init_history_motion
                initial_joints = torch.tensor(interaction['init_joints'], device=device,
                                              dtype=torch.float32)  # [3, 3]
                transf_rotmat, transf_transl = get_new_coordinate(initial_joints[None])
                transf_rotmat = transf_rotmat.repeat(batch_size, 1, 1)
                transf_transl = transf_transl.repeat(batch_size, 1, 1)

            motion_sequences, history_motion_tensor, transf_rotmat, transf_transl = optimize(
                history_motion_tensor, transf_rotmat, transf_transl, text_prompt, goal_joints, joints_mask)

            if all_motion_sequences is None:
                all_motion_sequences = motion_sequences
                all_motion_sequences['goal_location_list'] = [goal_joints[0, 0].cpu()]
                num_frames = all_motion_sequences['joints'].shape[1]
                all_motion_sequences['goal_location_idx'] = [0] * num_frames
            else:
                for key in motion_sequences:
                    if torch.is_tensor(motion_sequences[key]):
                        all_motion_sequences[key] = torch.cat([all_motion_sequences[key], motion_sequences[key]], dim=1)
                all_motion_sequences['texts'] += motion_sequences['texts']
                all_motion_sequences['goal_location_list'] += [goal_joints[0, 0].cpu()]
                num_goals = len(all_motion_sequences['goal_location_list'])
                num_frames = all_motion_sequences['joints'].shape[1]
                all_motion_sequences['goal_location_idx'] += [num_goals - 1] * num_frames
            with open(cache_path, 'wb') as f:
                pickle.dump([all_motion_sequences, history_motion_tensor, transf_rotmat, transf_transl], f)

    for idx in range(batch_size):
        sequence = {
            'texts': all_motion_sequences['texts'],
            'scene_path': scene_dir / 'scene_with_floor.obj',
            'goal_location_list': all_motion_sequences['goal_location_list'],
            'goal_location_idx': all_motion_sequences['goal_location_idx'],
            'gender': all_motion_sequences['gender'],
            'betas': all_motion_sequences['betas'][idx],
            'transl': all_motion_sequences['transl'][idx],
            'global_orient': all_motion_sequences['global_orient'][idx],
            'body_pose': all_motion_sequences['body_pose'][idx],
            'joints': all_motion_sequences['joints'][idx],
            'history_length': history_length,
            'future_length': future_length,
        }
        tensor_dict_to_device(sequence, 'cpu')
        with open(out_path / f'sample_{idx}.pkl', 'wb') as f:
            pickle.dump(sequence, f)

        # export smplx sequences for blender
        if optim_args.export_smpl:
            poses = transforms.matrix_to_axis_angle(
                torch.cat([sequence['global_orient'].reshape(-1, 1, 3, 3), sequence['body_pose']], dim=1)
            ).reshape(-1, 22 * 3)
            poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)],
                              dim=1)
            data_dict = {
                'mocap_framerate': dataset.target_fps,  # 30
                'gender': sequence['gender'],
                'betas': sequence['betas'][0, :10].detach().cpu().numpy(),
                'poses': poses.detach().cpu().numpy(),
                'trans': sequence['transl'].detach().cpu().numpy(),
            }
            with open(out_path / f'sample_{idx}_smplx.npz', 'wb') as f:
                np.savez(f, **data_dict)

    print(f'[Done] Results are at [{out_path.absolute()}]')
