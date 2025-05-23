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

debug = 0

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
    optim_steps: int = 300
    optim_unit_grad: int = 1
    optim_anneal_lr: int = 1
    weight_jerk: float = 0.0
    weight_collision: float = 1.0
    weight_contact: float = 0.0
    weight_skate: float = 0.0
    weight_self_collision: float = 0.1
    load_cache: int = 0
    contact_thresh: float = 0.03
    init_noise_scale: float = 1.0
    
    volsmpl_max_frames: int = 5
    volsmpl_model_folder: str = "./data/smplx_lockedhead_20230207/models_lockedhead"

    interaction_cfg: str = './data/optim_interaction/climb_up_stairs.json'


def calc_jerk(joints):
    vel = joints[:, 1:] - joints[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    jerk = acc[:, 1:] - acc[:, :-1]
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))
    jerk = jerk.amax(dim=[1, 2])
    return jerk.mean()

def create_scene_point_cloud(scene_mesh, num_points=8000, ensure_z_up=True, floor_height=0.0):
    mesh = scene_mesh.copy()
    
    if ensure_z_up:
        bounds = mesh.bounds
        min_z = bounds[0][2]
        if abs(min_z - floor_height) > 1e-5:
            translation = np.array([0, 0, floor_height - min_z])
            mesh.apply_translation(translation)
    
    points, face_indices = mesh.sample(num_points, return_index=True)
    face_normals = mesh.face_normals[face_indices]
    
    points_tensor = torch.tensor(points, dtype=torch.float32)
    normals_tensor = torch.tensor(face_normals, dtype=torch.float32)
    
    metadata = {
        'num_points': num_points,
        'bounds': mesh.bounds.tolist(),
        'center': mesh.centroid.tolist(),
        'floor_height': floor_height,
        'has_normals': True
    }
    
    return points_tensor, normals_tensor, metadata

def create_scene_assets(scene_dir, floor_height=0.0, num_scene_points=8000, device="cuda"):
    scene_with_floor_mesh = trimesh.load(scene_dir / 'scene_with_floor.obj', process=False, force='mesh')
    
    scene_points, scene_normals, scene_metadata = create_scene_point_cloud(
        scene_with_floor_mesh,
        num_points=num_scene_points,
        ensure_z_up=True,
        floor_height=floor_height
    )
    
    scene_points = scene_points.to(device)
    scene_normals = scene_normals.to(device)
    
    scene_assets = {
        'scene_with_floor_mesh': scene_with_floor_mesh,
        'floor_height': floor_height,
        'scene_points': scene_points,
        'scene_normals': scene_normals,
        'scene_metadata': scene_metadata
    }
    
    return scene_assets

class VolumetricSMPLCollisionCalculator:
    def __init__(self, model_folder, device="cuda"):
        self.model_folder = model_folder
        self.device = device
        self.smpl_model_cache = {}
        self.debug_step = 0
        
    def get_smpl_model(self, gender):
        if gender not in self.smpl_model_cache:
            import smplx
            from VolumetricSMPL import attach_volume
            
            model = smplx.create(
                self.model_folder, 
                model_type='smplx', 
                gender=gender, 
                device=self.device,
                dtype=torch.float32
            )
            model = model.to(self.device)
            attach_volume(model)
            
            if hasattr(model, 'volume'):
                model.volume = model.volume.to(self.device)
            
            self.smpl_model_cache[gender] = model
        
        return self.smpl_model_cache[gender]
    
    def analyze_human_scene_interaction(self, motion_sequences, scene_points, frame_indices):
        """分析人体-场景交互的核心信息"""
        joints = motion_sequences['joints']
        B, T, J, _ = joints.shape
        
        # 场景信息
        scene_min = scene_points.min(dim=0)[0]
        scene_max = scene_points.max(dim=0)[0]
        scene_center = scene_points.mean(dim=0)
        
        print(f"\n📊 人体-场景交互分析:")
        print(f"场景范围: X[{scene_min[0]:.2f}, {scene_max[0]:.2f}] Y[{scene_min[1]:.2f}, {scene_max[1]:.2f}] Z[{scene_min[2]:.2f}, {scene_max[2]:.2f}]")
        
        interaction_stats = {
            'penetration_frames': 0,
            'contact_frames': 0,
            'min_distances': [],
            'sdf_stats': []
        }
        
        # 分析选中的帧
        for i, frame_idx in enumerate(frame_indices):
            frame_joints = joints[0, frame_idx]  # [22, 3]
            pelvis_pos = frame_joints[0]
            
            # 计算人体到场景的最小距离
            distances = torch.cdist(frame_joints.unsqueeze(0), scene_points.unsqueeze(0))[0]  # [22, N]
            min_distance = distances.min().item()
            closest_joint_idx = distances.min(dim=1)[0].argmin().item()
            
            interaction_stats['min_distances'].append(min_distance)
            
            print(f"  帧{frame_idx:2d}: 骨盆({pelvis_pos[0]:.2f},{pelvis_pos[1]:.2f},{pelvis_pos[2]:.2f}) "
                  f"最近距离={min_distance:.3f}m 最近关节={closest_joint_idx}")
            
            if min_distance < 0.01:
                interaction_stats['contact_frames'] += 1
            if min_distance < -0.01:
                interaction_stats['penetration_frames'] += 1
        
        print(f"交互统计: 接触帧={interaction_stats['contact_frames']}/{len(frame_indices)} "
              f"穿透帧={interaction_stats['penetration_frames']}/{len(frame_indices)}")
        
        return interaction_stats
    
    def query_scene_points_with_volsmpl_sdf(self, motion_sequences, scene_points, frame_indices):
        gender = motion_sequences['gender']
        model = self.get_smpl_model(gender)
        
        B, T, _, _ = motion_sequences['joints'].shape
        num_scene_points = scene_points.shape[0]
        
        scene_points = scene_points.to(self.device)
        
        # 分析人体-场景交互
        interaction_stats = self.analyze_human_scene_interaction(motion_sequences, scene_points, frame_indices)
        
        sdf_tensor = torch.zeros(len(frame_indices), num_scene_points, device=self.device)
        total_collision_loss = torch.tensor(0.0, device=self.device)
        total_self_collision_loss = torch.tensor(0.0, device=self.device)
        
        sdf_summary = {'penetration_points': 0, 'contact_points': 0, 'free_points': 0}
        
        for i, frame_idx in enumerate(frame_indices):
            batch_idx = 0
            
            # 提取并转换SMPL参数
            global_orient_matrix = motion_sequences['global_orient'][batch_idx, frame_idx].detach().clone().to(device=self.device, dtype=torch.float32)
            body_pose_matrix = motion_sequences['body_pose'][batch_idx, frame_idx].detach().clone().to(device=self.device, dtype=torch.float32) 
            transl = motion_sequences['transl'][batch_idx, frame_idx].detach().clone().to(device=self.device, dtype=torch.float32)
            betas = motion_sequences['betas'][batch_idx, frame_idx, :10].detach().clone().to(device=self.device, dtype=torch.float32)
            
            global_orient_aa = transforms.matrix_to_axis_angle(global_orient_matrix).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            body_pose_aa = transforms.matrix_to_axis_angle(body_pose_matrix).to(device=self.device, dtype=torch.float32)
            body_pose_aa_flat = body_pose_aa.reshape(1, -1).to(device=self.device, dtype=torch.float32)
            transl_batch = transl.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            betas_batch = betas.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            
            # SMPL前向传播
            smpl_output = model(
                betas=betas_batch,
                global_orient=global_orient_aa,
                body_pose=body_pose_aa_flat,
                transl=transl_batch,
                return_verts=True,
                pose2rot=True
            )
            
            if hasattr(smpl_output, 'vertices'):
                smpl_output.vertices = smpl_output.vertices.to(self.device)
            if hasattr(smpl_output, 'joints'):
                smpl_output.joints = smpl_output.joints.to(self.device)
                
            full_pose = torch.cat([global_orient_aa, body_pose_aa_flat], dim=1).to(device=self.device, dtype=torch.float32)
            smpl_output.full_pose = full_pose
            
            # VolumetricSMPL SDF查询
            scene_points_batch = scene_points.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            
            try:
                sdf_values = model.volume.query(scene_points_batch, smpl_output)
                sdf_tensor[i] = sdf_values.squeeze(0).to(self.device)
                
                # SDF统计
                penetration = (sdf_values < -0.01).sum().item()
                contact = ((sdf_values >= -0.01) & (sdf_values <= 0.01)).sum().item()
                free = (sdf_values > 0.01).sum().item()
                
                sdf_summary['penetration_points'] += penetration
                sdf_summary['contact_points'] += contact
                sdf_summary['free_points'] += free
                
                if i == 0:  # 只在第一帧详细报告
                    print(f"  📈 SDF分析 帧{frame_idx}: 穿透{penetration} 接触{contact} 自由{free} "
                          f"范围[{sdf_values.min():.3f}, {sdf_values.max():.3f}]")
                
            except Exception as e:
                print(f"  ❌ SDF查询失败 帧{frame_idx}: {e}")
                sdf_tensor[i] = torch.zeros(num_scene_points, device=self.device)
            
            # 场景碰撞损失
            try:
                if not hasattr(smpl_output, 'full_pose'):
                    smpl_output.full_pose = full_pose
                
                scene_collision_loss = model.volume.collision_loss(scene_points_batch, smpl_output)
                
                if isinstance(scene_collision_loss, (list, tuple)):
                    scene_collision_loss = scene_collision_loss[0]
                
                if scene_collision_loss.dim() == 0:
                    scene_collision_loss = scene_collision_loss.unsqueeze(0)
                
                scene_collision_loss = scene_collision_loss.to(self.device)
                total_collision_loss += scene_collision_loss.mean()
                
            except Exception as e:
                if i == 0:
                    print(f"  ❌ 场景碰撞计算失败: {e}")
                scene_collision_loss = torch.tensor(0.0, device=self.device)
                total_collision_loss += scene_collision_loss
            
            # 自碰撞损失
            try:
                if not hasattr(smpl_output, 'full_pose'):
                    smpl_output.full_pose = full_pose
                    
                self_collision_loss = model.volume.self_collision_loss(smpl_output)
                
                if isinstance(self_collision_loss, (list, tuple)):
                    self_collision_loss = self_collision_loss[0]
                
                if self_collision_loss.dim() == 0:
                    self_collision_loss = self_collision_loss.unsqueeze(0)
                elif self_collision_loss.dim() > 1:
                    self_collision_loss = self_collision_loss.mean()
                    
                self_collision_loss = self_collision_loss.to(self.device)
                total_self_collision_loss += self_collision_loss.mean()
                
            except Exception as e:
                if i == 0:
                    print(f"  ❌ 自碰撞计算失败: {e}")
                self_collision_loss = torch.tensor(0.0, device=self.device)
                total_self_collision_loss += self_collision_loss
        
        # 最终统计
        num_frames = len(frame_indices)
        num_total_points = num_frames * num_scene_points
        
        avg_collision_loss = (total_collision_loss.mean() / num_frames).squeeze()
        avg_self_collision_loss = (total_self_collision_loss.mean() / num_frames).squeeze()
        
        print(f"🎯 SDF总结: 穿透{sdf_summary['penetration_points']}/{num_total_points}点 "
              f"接触{sdf_summary['contact_points']}/{num_total_points}点 "
              f"自由{sdf_summary['free_points']}/{num_total_points}点")
        print(f"💥 损失: 场景碰撞={avg_collision_loss.item():.6f} 自碰撞={avg_self_collision_loss.item():.6f}")
        
        return sdf_tensor, avg_collision_loss, avg_self_collision_loss
    
    def calculate_contact_loss(self, motion_sequences, scene_points, frame_indices, contact_thresh=0.03):
        try:
            scene_points = scene_points.to(self.device)
            sdf_tensor, _, _ = self.query_scene_points_with_volsmpl_sdf(
                motion_sequences, scene_points, frame_indices
            )
            
            contact_loss = torch.relu(contact_thresh - torch.abs(sdf_tensor)).mean()
            contact_loss = contact_loss.to(self.device)
            
            return contact_loss
        except Exception as e:
            print(f"❌ 接触损失计算失败: {e}")
            return torch.tensor(0.0, device=self.device)

def optimize(history_motion_tensor, transf_rotmat, transf_transl, text_prompt, goal_joints, joints_mask):
    texts = []
    if ',' in text_prompt:
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
            text_embedding = all_text_embedding[segment_id].expand(batch_size, -1)
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
                skip_timesteps=0,
                init_image=None,
                progress=False,
                noise=noise[segment_id],
            )

            latent_pred = x_start_pred.permute(1, 0, 2)
            future_motion_pred = vae_model.decode(latent_pred, history_motion, nfuture=future_length,
                                                       scale_latent=denoiser_args.rescale_latent)

            future_frames = dataset.denormalize(future_motion_pred)
            new_history_frames = future_frames[:, -history_length:, :]

            if segment_id == 0:
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
                    motion_sequences[key] = torch.cat([motion_sequences[key], future_primitive_dict[key]], dim=1)

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
            history_motion = dataset.normalize(history_motion)

        return motion_sequences, history_motion, transf_rotmat, transf_transl
        
    volsmpl_calculator = VolumetricSMPLCollisionCalculator(
        model_folder=optim_args.volsmpl_model_folder,
        device=device
    )
        
    optim_steps = optim_args.optim_steps
    lr = optim_args.optim_lr
    
    noise = torch.randn(num_rollout, batch_size, *denoiser_args.model_args.noise_shape,
                        device=device, dtype=torch.float32)
    noise = noise * optim_args.init_noise_scale
    noise.requires_grad_(True)
    
    reduction_dims = list(range(1, len(noise.shape)))
    criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)

    optimizer = torch.optim.Adam([noise], lr=lr)
    
    print(f"\n🚀 开始VolumetricSMPL优化: {optim_steps}步, 学习率{lr}")
    print("="*80)
    
    for i in tqdm(range(optim_steps), desc="优化进度"):
        optimizer.zero_grad()
        if optim_args.optim_anneal_lr:
            frac = 1.0 - i / optim_steps
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow

        motion_sequences, new_history_motion_tensor, new_transf_rotmat, new_transf_transl = rollout(noise,
                                                                                                    history_motion_tensor,
                                                                                                    transf_rotmat,
                                                                                                    transf_transl)

        B, T, _, _ = motion_sequences['joints'].shape
        
        for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
            if torch.is_tensor(motion_sequences[key]):
                motion_sequences[key] = motion_sequences[key].to(device)
        
        max_frames = min(optim_args.volsmpl_max_frames, T)
        frame_indices = torch.linspace(0, T-1, max_frames, dtype=torch.long)
        
        scene_points = scene_assets['scene_points'].to(device)
        
        try:
            if i % 20 == 0:  # 每20步详细分析一次
                print(f"\n🔍 第{i}步详细分析:")
            
            sdf_tensor, collision_loss, self_collision_loss = volsmpl_calculator.query_scene_points_with_volsmpl_sdf(
                motion_sequences, scene_points, frame_indices
            )
            
            contact_loss = volsmpl_calculator.calculate_contact_loss(
                motion_sequences, scene_points, frame_indices, optim_args.contact_thresh
            )
            
            collision_loss = collision_loss.to(device)
            self_collision_loss = self_collision_loss.to(device)
            contact_loss = contact_loss.to(device)
            
            if collision_loss.dim() > 0:
                collision_loss = collision_loss.mean()
            if self_collision_loss.dim() > 0:
                self_collision_loss = self_collision_loss.mean()
            if contact_loss.dim() > 0:
                contact_loss = contact_loss.mean()
            
        except Exception as e:
            if i % 20 == 0:
                print(f"❌ VolumetricSMPL计算失败: {e}")
            collision_loss = torch.tensor(0.0, device=device)
            self_collision_loss = torch.tensor(0.0, device=device)
            contact_loss = torch.tensor(0.0, device=device)
        
        goal_joints_gpu = goal_joints.to(device)
        joints_mask_gpu = joints_mask.to(device)
        
        loss_joints = criterion(
            motion_sequences['joints'][:, -1, joints_mask_gpu], 
            goal_joints_gpu[:, joints_mask_gpu]
        )
        loss_jerk = calc_jerk(motion_sequences['joints'])
        
        loss_joints = loss_joints.to(device)
        loss_jerk = loss_jerk.to(device)
        
        total_loss = (
            loss_joints +
            optim_args.weight_collision * collision_loss +
            optim_args.weight_self_collision * self_collision_loss +
            optim_args.weight_contact * contact_loss +
            optim_args.weight_jerk * loss_jerk
        )
        
        total_loss = total_loss.to(device)

        total_loss.backward()
        if optim_args.optim_unit_grad:
            noise.grad.data /= noise.grad.norm(p=2, dim=reduction_dims, keepdim=True).clamp(min=1e-6)
        optimizer.step()
        
        # 简洁的进度报告
        if i % 20 == 0:
            print(f'📊 [{i:3d}/{optim_steps}] 总损失={total_loss.item():.4f} | '
                  f'目标={loss_joints.item():.4f} 场景碰撞={collision_loss.item():.4f} '
                  f'自碰撞={self_collision_loss.item():.4f} 接触={contact_loss.item():.4f} '
                  f'平滑={loss_jerk.item():.4f}')

    for key in motion_sequences:
        if torch.is_tensor(motion_sequences[key]):
            motion_sequences[key] = motion_sequences[key].detach()
    motion_sequences['texts'] = texts
    return motion_sequences, new_history_motion_tensor.detach(), new_transf_rotmat.detach(), new_transf_transl.detach()

if __name__ == '__main__':
    optim_args = tyro.cli(OptimArgs)
    
    if optim_args.batch_size != 1:
        print(f"⚠️ VolumetricSMPL强制设置batch_size=1")
        optim_args.batch_size = 1
    
    random.seed(optim_args.seed)
    np.random.seed(optim_args.seed)
    torch.manual_seed(optim_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = optim_args.torch_deterministic
    device = torch.device(optim_args.device if torch.cuda.is_available() else "cpu")
    optim_args.device = device

    print("🚀 初始化VolumetricSMPL系统")
    print(f"设备: {device} | 优化步数: {optim_args.optim_steps}")
    print(f"权重: 碰撞={optim_args.weight_collision} 自碰撞={optim_args.weight_self_collision} 接触={optim_args.weight_contact}")

    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(optim_args.denoiser_checkpoint, device)
    denoiser_checkpoint = Path(optim_args.denoiser_checkpoint)
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / 'volsmpl_optim'
    save_dir.mkdir(parents=True, exist_ok=True)
    optim_args.save_dir = save_dir

    diffusion_args = denoiser_args.diffusion_args
    assert 'ddim' in optim_args.respacing
    diffusion_args.respacing = optim_args.respacing
    diffusion = create_gaussian_diffusion(diffusion_args)
    sample_fn = diffusion.ddim_sample_loop_full_chain

    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,
                                     dataset_path=vae_args.data_args.data_dir,
                                     sequence_path='./data/stand.pkl',
                                     batch_size=1,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     )
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    primitive_utility = dataset.primitive_utility
    batch_size = 1

    with open(optim_args.interaction_cfg, 'r') as f:
        interaction_cfg = json.load(f)
    interaction_name = interaction_cfg['interaction_name'].replace(' ', '_')
    scene_dir = Path(interaction_cfg['scene_dir'])
    scene_dir = Path(scene_dir)
    
    scene_assets = create_scene_assets(
        scene_dir=scene_dir,
        floor_height=interaction_cfg['floor_height'],
        num_scene_points=8000,
        device=device
    )
    
    print(f"🏢 场景加载完成: {scene_assets['scene_points'].shape[0]}点")

    out_path = optim_args.save_dir
    filename = f'volsmpl_analysis_guidance{optim_args.guidance_param}_seed{optim_args.seed}'
    if optim_args.respacing != '':
        filename = f'{optim_args.respacing}_{filename}'
    if optim_args.zero_noise:
        filename = f'zero_noise_{filename}'
    if optim_args.use_predicted_joints:
        filename = f'use_pred_joints_{filename}'
    filename = f'{interaction_name}_{filename}'
    filename = f'{filename}_collision{optim_args.weight_collision}_self{optim_args.weight_self_collision}_contact{optim_args.weight_contact}_jerk{optim_args.weight_jerk}_frames{optim_args.volsmpl_max_frames}'
    
    out_path = out_path / filename
    out_path.mkdir(parents=True, exist_ok=True)

    batch = dataset.get_batch(batch_size=optim_args.batch_size)
    input_motions, model_kwargs = batch[0]['motion_tensor_normalized'], {'y': batch[0]}
    del model_kwargs['y']['motion_tensor_normalized']
    gender = model_kwargs['y']['gender'][0]
    betas = model_kwargs['y']['betas'][:, :primitive_length, :].to(device)
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': betas[:, 0, :],
        'gender': gender,
    })
    input_motions = input_motions.to(device)
    motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)
    init_history_motion = motion_tensor[:, :history_length, :]

    all_motion_sequences = None
    for interaction_idx, interaction in enumerate(interaction_cfg['interactions']):
        cache_path = out_path / f'volsmpl_analysis_cache_{interaction_idx}.pkl'
        if cache_path.exists() and optim_args.load_cache:
            with open(cache_path, 'rb') as f:
                all_motion_sequences, history_motion_tensor, transf_rotmat, transf_transl = pickle.load(f)
            tensor_dict_to_device(all_motion_sequences, device)
            history_motion_tensor = history_motion_tensor.to(device)
            transf_rotmat = transf_rotmat.to(device)
            transf_transl = transf_transl.to(device)
            print(f"📂 从缓存加载交互{interaction_idx}")
        else:
            text_prompt = interaction['text_prompt']
            
            goal_joints = torch.zeros(batch_size, 22, 3, device=device, dtype=torch.float32)
            goal_joints[:, 0] = torch.tensor(interaction['goal_joints'][0], device=device, dtype=torch.float32)
            joints_mask = torch.zeros(22, device=device, dtype=torch.bool)
            joints_mask[0] = 1

            if interaction_idx == 0:
                history_motion_tensor = init_history_motion
                initial_joints = torch.tensor(interaction['init_joints'], device=device,
                                              dtype=torch.float32)
                transf_rotmat, transf_transl = get_new_coordinate(initial_joints[None])
                transf_rotmat = transf_rotmat.repeat(batch_size, 1, 1).to(device)
                transf_transl = transf_transl.repeat(batch_size, 1, 1).to(device)

            print(f"\n🎯 开始交互 {interaction_idx+1}: {text_prompt}")
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
        with open(out_path / f'volsmpl_analysis_sample_{idx}.pkl', 'wb') as f:
            pickle.dump(sequence, f)

        if optim_args.export_smpl:
            poses = transforms.matrix_to_axis_angle(
                torch.cat([sequence['global_orient'].reshape(-1, 1, 3, 3), sequence['body_pose']], dim=1)
            ).reshape(-1, 22 * 3)
            poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)],
                              dim=1)
            data_dict = {
                'mocap_framerate': dataset.target_fps,
                'gender': sequence['gender'],
                'betas': sequence['betas'][0, :10].detach().cpu().numpy(),
                'poses': poses.detach().cpu().numpy(),
                'trans': sequence['transl'].detach().cpu().numpy(),
            }
            with open(out_path / f'volsmpl_analysis_sample_{idx}_smplx.npz', 'wb') as f:
                np.savez(f, **data_dict)

    print(f'✅ 分析完成! 结果保存在: {out_path.absolute()}')
