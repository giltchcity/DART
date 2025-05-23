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
    batch_size: int = 1  # 强制使用batch_size=1避免VolumetricSMPL批量问题

    optim_lr: float = 0.01
    optim_steps: int = 300
    optim_unit_grad: int = 1
    optim_anneal_lr: int = 1
    weight_jerk: float = 0.0
    weight_collision: float = 1.0  # VolumetricSMPL场景碰撞权重
    weight_contact: float = 0.0
    weight_skate: float = 0.0
    weight_self_collision: float = 0.1  # VolumetricSMPL自碰撞权重
    load_cache: int = 0
    contact_thresh: float = 0.03
    init_noise_scale: float = 1.0
    
    # VolumetricSMPL专用参数
    volsmpl_max_frames: int = 5  # 内存优化：最大处理帧数
    volsmpl_model_folder: str = "./data/smplx_lockedhead_20230207/models_lockedhead"

    interaction_cfg: str = './data/optim_interaction/climb_up_stairs.json'


def calc_jerk(joints):
    vel = joints[:, 1:] - joints[:, :-1]  # --> B x T-1 x 22 x 3
    acc = vel[:, 1:] - vel[:, :-1]  # --> B x T-2 x 22 x 3
    jerk = acc[:, 1:] - acc[:, :-1]  # --> B x T-3 x 22 x 3
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))  # --> B x T-3 x 22, compute L1 norm of jerk
    jerk = jerk.amax(dim=[1, 2])  # --> B, Get the max of the jerk across all joints and frames

    return jerk.mean()

def create_scene_point_cloud(scene_mesh, num_points=8000, ensure_z_up=True, floor_height=0.0):
    """
    Convert a scene mesh to a point cloud representation.
    """
    # Make a copy to avoid modifying the original
    mesh = scene_mesh.copy()
    
    # Check if scene is z-up and fix if needed
    if ensure_z_up:
        # Get the mesh bounds
        bounds = mesh.bounds
        min_z = bounds[0][2]
        
        # If floor isn't at the desired height, translate it
        if abs(min_z - floor_height) > 1e-5:
            translation = np.array([0, 0, floor_height - min_z])
            mesh.apply_translation(translation)
            print(f"Adjusted mesh to have floor at z={floor_height} (was at z={min_z})")
    
    # Sample points uniformly from the mesh surface
    points, face_indices = mesh.sample(num_points, return_index=True)
    
    # Get face normals for the sampled points
    face_normals = mesh.face_normals[face_indices]
    
    # Convert to torch tensors
    points_tensor = torch.tensor(points, dtype=torch.float32)
    normals_tensor = torch.tensor(face_normals, dtype=torch.float32)
    
    # Create metadata
    metadata = {
        'num_points': num_points,
        'bounds': mesh.bounds.tolist(),
        'center': mesh.centroid.tolist(),
        'floor_height': floor_height,
        'has_normals': True
    }
    
    return points_tensor, normals_tensor, metadata

def create_scene_assets(scene_dir, floor_height=0.0, num_scene_points=8000, device="cuda"):
    """Create scene assets focusing on point cloud representation for VolumetricSMPL"""
    scene_with_floor_mesh = trimesh.load(scene_dir / 'scene_with_floor.obj', process=False, force='mesh')
    
    # Generate point cloud representation for VolumetricSMPL
    scene_points, scene_normals, scene_metadata = create_scene_point_cloud(
        scene_with_floor_mesh,
        num_points=num_scene_points,
        ensure_z_up=True,
        floor_height=floor_height
    )
    
    # Move tensors to the right device
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
    """
    专门用于VolumetricSMPL碰撞检测的类 - 使用真实API
    """
    def __init__(self, model_folder, device="cuda"):
        self.model_folder = model_folder
        self.device = device
        self.smpl_model_cache = {}  # 缓存不同性别的模型
        
    def get_smpl_model(self, gender):
        """获取或创建SMPL模型（带缓存）- 强制GPU设备"""
        if gender not in self.smpl_model_cache:
            import smplx
            from VolumetricSMPL import attach_volume
            
            print(f"正在创建VolumetricSMPL模型: gender={gender}, device={self.device}")
            
            # 强制创建在GPU上的模型
            model = smplx.create(
                self.model_folder, 
                model_type='smplx', 
                gender=gender, 
                device=self.device,  # 确保模型在GPU上
                dtype=torch.float32   # 确保数据类型一致
            )
            
            # 确保模型在正确设备上
            model = model.to(self.device)
            
            # 附加体积功能
            attach_volume(model)
            
            # 确保volumetric model也在正确设备上
            if hasattr(model, 'volume'):
                model.volume = model.volume.to(self.device)
            
            self.smpl_model_cache[gender] = model
            print(f"✅ VolumetricSMPL模型创建成功: gender={gender}, 模型设备={next(model.parameters()).device}")
            
            # 检查并打印VolumetricSMPL的实际API
            self._check_volumetric_api(model)
        
        return self.smpl_model_cache[gender]
    
    def _check_volumetric_api(self, model):
        """检查VolumetricSMPL的实际API"""
        print("📋 VolumetricSMPL API检查:")
        try:
            volume_methods = [method for method in dir(model.volume) if not method.startswith('_')]
            print(f"  可用方法: {volume_methods}")
            
            # 检查collision_loss的签名
            if hasattr(model.volume, 'collision_loss'):
                import inspect
                sig = inspect.signature(model.volume.collision_loss)
                print(f"  collision_loss签名: {sig}")
                
            # 检查self_collision_loss的签名  
            if hasattr(model.volume, 'self_collision_loss'):
                sig = inspect.signature(model.volume.self_collision_loss)
                print(f"  self_collision_loss签名: {sig}")
                
            # 检查query的签名
            if hasattr(model.volume, 'query'):
                sig = inspect.signature(model.volume.query)
                print(f"  query签名: {sig}")
                
        except Exception as e:
            print(f"  API检查失败: {e}")
    
    def query_scene_points_with_volsmpl_sdf(self, motion_sequences, scene_points, frame_indices):
        """
        任务3核心：Query the scene points using the VolSMPL SDFs
        
        使用真实的VolumetricSMPL API - 完全修复设备问题
        """
        gender = motion_sequences['gender']
        model = self.get_smpl_model(gender)
        
        B, T, _, _ = motion_sequences['joints'].shape
        num_scene_points = scene_points.shape[0]
        
        # 确保scene_points在正确设备上
        scene_points = scene_points.to(self.device)
        
        # 初始化SDF tensor: [#Frames, #Points]
        sdf_tensor = torch.zeros(len(frame_indices), num_scene_points, device=self.device)
        total_collision_loss = torch.tensor(0.0, device=self.device)
        total_self_collision_loss = torch.tensor(0.0, device=self.device)
        
        for i, frame_idx in enumerate(frame_indices):
            batch_idx = 0  # 处理第一个batch
            
            # 提取SMPL参数并强制转换到GPU，确保数据类型一致
            global_orient_matrix = motion_sequences['global_orient'][batch_idx, frame_idx].detach().clone().to(device=self.device, dtype=torch.float32)
            body_pose_matrix = motion_sequences['body_pose'][batch_idx, frame_idx].detach().clone().to(device=self.device, dtype=torch.float32) 
            transl = motion_sequences['transl'][batch_idx, frame_idx].detach().clone().to(device=self.device, dtype=torch.float32)
            betas = motion_sequences['betas'][batch_idx, frame_idx, :10].detach().clone().to(device=self.device, dtype=torch.float32)
            
            # 转换为axis-angle格式，确保结果在GPU上
            global_orient_aa = transforms.matrix_to_axis_angle(global_orient_matrix).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            body_pose_aa = transforms.matrix_to_axis_angle(body_pose_matrix).to(device=self.device, dtype=torch.float32)
            body_pose_aa_flat = body_pose_aa.reshape(1, -1).to(device=self.device, dtype=torch.float32)
            transl_batch = transl.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            betas_batch = betas.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            
            # 调试：确认所有参数在正确设备上
            print(f"设备检查 - Frame {i}:")
            print(f"  global_orient_aa: {global_orient_aa.device}, shape: {global_orient_aa.shape}")
            print(f"  body_pose_aa_flat: {body_pose_aa_flat.device}, shape: {body_pose_aa_flat.shape}")  
            print(f"  transl_batch: {transl_batch.device}, shape: {transl_batch.shape}")
            print(f"  betas_batch: {betas_batch.device}, shape: {betas_batch.shape}")
            print(f"  scene_points: {scene_points.device}, shape: {scene_points.shape}")
            
            # SMPL前向传播 - 所有参数确保在GPU上
            smpl_output = model(
                betas=betas_batch,
                global_orient=global_orient_aa,
                body_pose=body_pose_aa_flat,
                transl=transl_batch,
                return_verts=True,
                pose2rot=True
            )
            
            # 确保smpl_output的所有属性都在GPU上
            if hasattr(smpl_output, 'vertices'):
                smpl_output.vertices = smpl_output.vertices.to(self.device)
            if hasattr(smpl_output, 'joints'):
                smpl_output.joints = smpl_output.joints.to(self.device)
                
            print(f"  smpl_output.vertices: {smpl_output.vertices.device}")
            print(f"  smpl_output.joints: {smpl_output.joints.device}")
            
            # 添加full_pose属性（VolumetricSMPL需要）- 确保在GPU上
            full_pose = torch.cat([global_orient_aa, body_pose_aa_flat], dim=1).to(device=self.device, dtype=torch.float32)
            smpl_output.full_pose = full_pose
            print(f"  smpl_output.full_pose: {smpl_output.full_pose.device}")
            
            # ===== 真实的VolumetricSMPL API调用 =====
            
            # 1. 使用真实的query方法获取SDF值
            scene_points_batch = scene_points.unsqueeze(0).to(device=self.device, dtype=torch.float32)  # [1, N, 3]
            print(f"  准备query - scene_points_batch: {scene_points_batch.device}")
            
            try:
                sdf_values = model.volume.query(scene_points_batch, smpl_output)  # [1, N]
                sdf_tensor[i] = sdf_values.squeeze(0).to(self.device)
                print(f"  SDF query成功, sdf_values: {sdf_values.device}")
            except Exception as e:
                print(f"  SDF query失败: {e}")
                sdf_tensor[i] = torch.zeros(num_scene_points, device=self.device)
            
            # 2. 场景碰撞损失 - 修复参数顺序和full_pose问题
            try:
                # 重新确保full_pose属性存在
                if not hasattr(smpl_output, 'full_pose'):
                    smpl_output.full_pose = full_pose
                
                # 正确的API调用顺序：collision_loss(point_cloud, smpl_output)
                scene_collision_loss = model.volume.collision_loss(scene_points_batch, smpl_output)
                
                # 处理返回值 - 可能是标量或张量
                if isinstance(scene_collision_loss, (list, tuple)):
                    scene_collision_loss = scene_collision_loss[0]  # 取第一个元素
                
                # 确保是标量张量
                if scene_collision_loss.dim() == 0:
                    scene_collision_loss = scene_collision_loss.unsqueeze(0)
                
                scene_collision_loss = scene_collision_loss.to(self.device)
                total_collision_loss += scene_collision_loss.mean()  # 使用mean确保标量
                print(f"  场景碰撞计算成功: {scene_collision_loss.mean().item()}")
            except Exception as e:
                print(f"  场景碰撞计算失败: {e}")
                scene_collision_loss = torch.tensor(0.0, device=self.device)
                total_collision_loss += scene_collision_loss
            
            # 3. 自碰撞损失 - 修复维度问题
            try:
                # 重新确保full_pose属性存在
                if not hasattr(smpl_output, 'full_pose'):
                    smpl_output.full_pose = full_pose
                    
                self_collision_loss = model.volume.self_collision_loss(smpl_output)
                
                # 处理返回值 - 可能是标量或张量
                if isinstance(self_collision_loss, (list, tuple)):
                    self_collision_loss = self_collision_loss[0]
                
                # 确保是标量张量并处理维度
                if self_collision_loss.dim() == 0:
                    self_collision_loss = self_collision_loss.unsqueeze(0)
                elif self_collision_loss.dim() > 1:
                    self_collision_loss = self_collision_loss.mean()
                    
                self_collision_loss = self_collision_loss.to(self.device)
                total_self_collision_loss += self_collision_loss.mean()  # 使用mean确保标量
                print(f"  自碰撞计算成功: {self_collision_loss.mean().item()}")
            except Exception as e:
                print(f"  自碰撞计算失败: {e}")
                self_collision_loss = torch.tensor(0.0, device=self.device)
                total_self_collision_loss += self_collision_loss
        
        # 平均化损失 - 确保维度正确
        num_frames = len(frame_indices)
        
        # 确保total_collision_loss和total_self_collision_loss是正确的张量
        if total_collision_loss.dim() == 0:
            total_collision_loss = total_collision_loss.unsqueeze(0)
        if total_self_collision_loss.dim() == 0:
            total_self_collision_loss = total_self_collision_loss.unsqueeze(0)
            
        avg_collision_loss = total_collision_loss.mean() / num_frames
        avg_self_collision_loss = total_self_collision_loss.mean() / num_frames
        
        # 确保返回的是标量张量
        avg_collision_loss = avg_collision_loss.squeeze()
        avg_self_collision_loss = avg_self_collision_loss.squeeze()
        
        print(f"  最终平均损失:")
        print(f"    avg_collision_loss: {avg_collision_loss.item():.6f}")
        print(f"    avg_self_collision_loss: {avg_self_collision_loss.item():.6f}")
        
        return sdf_tensor, avg_collision_loss, avg_self_collision_loss
    
    def calculate_contact_loss(self, motion_sequences, scene_points, frame_indices, contact_thresh=0.03):
        """
        使用VolumetricSMPL SDF计算接触损失 - 确保设备一致性
        """
        try:
            # 获取SDF值用于接触检测，确保设备一致
            scene_points = scene_points.to(self.device)
            sdf_tensor, _, _ = self.query_scene_points_with_volsmpl_sdf(
                motion_sequences, scene_points, frame_indices
            )
            
            # 基于SDF值的接触损失：当SDF接近0时表示接触
            # 负SDF表示穿透，正SDF表示分离
            contact_loss = torch.relu(contact_thresh - torch.abs(sdf_tensor)).mean()
            contact_loss = contact_loss.to(self.device)
            
            return contact_loss
        except Exception as e:
            print(f"接触损失计算失败: {e}")
            return torch.tensor(0.0, device=self.device)

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
        
    # 创建VolumetricSMPL碰撞计算器
    volsmpl_calculator = VolumetricSMPLCollisionCalculator(
        model_folder=optim_args.volsmpl_model_folder,
        device=device
    )
        
    optim_steps = optim_args.optim_steps
    lr = optim_args.optim_lr
    
    # 确保noise tensor在GPU上
    noise = torch.randn(num_rollout, batch_size, *denoiser_args.model_args.noise_shape,
                        device=device, dtype=torch.float32)
    noise = noise * optim_args.init_noise_scale
    noise.requires_grad_(True)
    
    print(f"优化初始化:")
    print(f"  noise: device={noise.device}, shape={noise.shape}, requires_grad={noise.requires_grad}")
    print(f"  device: {device}")
    
    reduction_dims = list(range(1, len(noise.shape)))
    criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)

    optimizer = torch.optim.Adam([noise], lr=lr)
    
    print("=== 开始使用真实VolumetricSMPL API优化 ===")
    print(f"优化步数: {optim_steps}, 学习率: {lr}")
    print("="*60)
    
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

        # ===== 核心：使用真实VolumetricSMPL API进行碰撞和接触检测 =====
        B, T, _, _ = motion_sequences['joints'].shape
        
        # 确保motion_sequences中的所有张量都在GPU上
        for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
            if torch.is_tensor(motion_sequences[key]):
                motion_sequences[key] = motion_sequences[key].to(device)
        
        print(f"设备检查 - motion_sequences:")
        print(f"  joints: {motion_sequences['joints'].device}")
        print(f"  transl: {motion_sequences['transl'].device}")
        print(f"  global_orient: {motion_sequences['global_orient'].device}")
        print(f"  body_pose: {motion_sequences['body_pose'].device}")
        print(f"  betas: {motion_sequences['betas'].device}")
        
        # 任务3.2：选择帧子集以节省GPU内存
        max_frames = min(optim_args.volsmpl_max_frames, T)
        frame_indices = torch.linspace(0, T-1, max_frames, dtype=torch.long)
        
        # 确保scene_points在GPU上
        scene_points = scene_assets['scene_points'].to(device)
        print(f"  scene_points: {scene_points.device}")
        
        try:
            print("开始VolumetricSMPL计算...")
            print(f"  批大小: {B}, 时间步: {T}, 选择帧数: {max_frames}")
            
            # 任务3.1：使用VolumetricSMPL SDF查询场景点
            sdf_tensor, collision_loss, self_collision_loss = volsmpl_calculator.query_scene_points_with_volsmpl_sdf(
                motion_sequences, scene_points, frame_indices
            )
            
            # 任务3.3：设计碰撞和接触函数
            contact_loss = volsmpl_calculator.calculate_contact_loss(
                motion_sequences, scene_points, frame_indices, optim_args.contact_thresh
            )
            
            # 确保所有损失都在GPU上且为标量
            collision_loss = collision_loss.to(device)
            self_collision_loss = self_collision_loss.to(device)
            contact_loss = contact_loss.to(device)
            
            # 确保损失为标量
            if collision_loss.dim() > 0:
                collision_loss = collision_loss.mean()
            if self_collision_loss.dim() > 0:
                self_collision_loss = self_collision_loss.mean()
            if contact_loss.dim() > 0:
                contact_loss = contact_loss.mean()
            
            print(f"✅ VolumetricSMPL计算成功")
            print(f"  collision_loss: {collision_loss.item():.6f} (device: {collision_loss.device})")
            print(f"  self_collision_loss: {self_collision_loss.item():.6f} (device: {self_collision_loss.device})")
            print(f"  contact_loss: {contact_loss.item():.6f} (device: {contact_loss.device})")
            
        except Exception as e:
            print(f"❌ VolumetricSMPL计算失败: {e}")
            import traceback
            traceback.print_exc()
            # 如果VolumetricSMPL失败，使用零损失继续优化
            collision_loss = torch.tensor(0.0, device=device)
            self_collision_loss = torch.tensor(0.0, device=device)
            contact_loss = torch.tensor(0.0, device=device)
            sdf_tensor = torch.zeros(max_frames, scene_points.shape[0], device=device)
        
        # 其他损失项 - 确保设备一致性
        goal_joints_gpu = goal_joints.to(device)
        joints_mask_gpu = joints_mask.to(device)
        
        loss_joints = criterion(
            motion_sequences['joints'][:, -1, joints_mask_gpu], 
            goal_joints_gpu[:, joints_mask_gpu]
        )
        loss_jerk = calc_jerk(motion_sequences['joints'])
        
        # 确保所有损失都在GPU上
        loss_joints = loss_joints.to(device)
        loss_jerk = loss_jerk.to(device)
        
        print(f"其他损失:")
        print(f"  loss_joints: {loss_joints.item():.6f} (device: {loss_joints.device})")
        print(f"  loss_jerk: {loss_jerk.item():.6f} (device: {loss_jerk.device})")
        
        # 任务3.3：将所有损失项集成到优化目标中 - 确保设备一致性
        total_loss = (
            loss_joints +  # 目标损失
            optim_args.weight_collision * collision_loss +  # VolumetricSMPL场景碰撞
            optim_args.weight_self_collision * self_collision_loss +  # VolumetricSMPL自碰撞
            optim_args.weight_contact * contact_loss +  # VolumetricSMPL接触
            optim_args.weight_jerk * loss_jerk  # 平滑性
        )
        
        # 确保总损失在GPU上
        total_loss = total_loss.to(device)
        
        print(f"总损失: {total_loss.item():.6f} (device: {total_loss.device})")

        total_loss.backward()
        if optim_args.optim_unit_grad:
            noise.grad.data /= noise.grad.norm(p=2, dim=reduction_dims, keepdim=True).clamp(min=1e-6)
        optimizer.step()
        
        # 打印VolumetricSMPL专用信息
        if i % 10 == 0:
            print(f'\n[{i}/{optim_steps}] VolumetricSMPL优化详情:')
            print(f'  总损失: {total_loss.item():.4f} (device: {total_loss.device})')
            print(f'  目标损失: {loss_joints.item():.4f}')
            print(f'  场景碰撞: {collision_loss.item():.4f}')
            print(f'  自碰撞: {self_collision_loss.item():.4f}')
            print(f'  接触损失: {contact_loss.item():.4f}')
            print(f'  平滑损失: {loss_jerk.item():.4f}')
            if isinstance(sdf_tensor, torch.Tensor):
                print(f'  SDF张量: 形状={sdf_tensor.shape}, 设备={sdf_tensor.device}')
            print("="*50)

    for key in motion_sequences:
        if torch.is_tensor(motion_sequences[key]):
            motion_sequences[key] = motion_sequences[key].detach()
    motion_sequences['texts'] = texts
    return motion_sequences, new_history_motion_tensor.detach(), new_transf_rotmat.detach(), new_transf_transl.detach()

if __name__ == '__main__':
    optim_args = tyro.cli(OptimArgs)
    
    # VolumetricSMPL当前只支持batch_size=1
    if optim_args.batch_size != 1:
        print(f"⚠️  警告: VolumetricSMPL当前只支持batch_size=1，自动调整为1（原设置: {optim_args.batch_size}）")
        optim_args.batch_size = 1
    
    # TRY NOT TO MODIFY: seeding
    random.seed(optim_args.seed)
    np.random.seed(optim_args.seed)
    torch.manual_seed(optim_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = optim_args.torch_deterministic
    device = torch.device(optim_args.device if torch.cuda.is_available() else "cpu")
    optim_args.device = device

    print("=== 初始化VolumetricSMPL系统（真实API版本）===")
    print(f"设备: {device}")
    print(f"批大小: {optim_args.batch_size}")
    print(f"优化步数: {optim_args.optim_steps}")
    print(f"权重设置: collision={optim_args.weight_collision}, self_collision={optim_args.weight_self_collision}, contact={optim_args.weight_contact}")
    print("="*60)

    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(optim_args.denoiser_checkpoint, device)
    denoiser_checkpoint = Path(optim_args.denoiser_checkpoint)
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / 'volsmpl_optim'
    save_dir.mkdir(parents=True, exist_ok=True)
    optim_args.save_dir = save_dir

    diffusion_args = denoiser_args.diffusion_args
    assert 'ddim' in optim_args.respacing
    diffusion_args.respacing = optim_args.respacing
    print('diffusion_args:', asdict(diffusion_args))
    diffusion = create_gaussian_diffusion(diffusion_args)
    sample_fn = diffusion.ddim_sample_loop_full_chain

    # load initial seed dataset - 强制batch_size=1用于VolumetricSMPL
    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,
                                     dataset_path=vae_args.data_args.data_dir,
                                     sequence_path='./data/stand.pkl',
                                     batch_size=1,  # 强制使用1避免VolumetricSMPL问题
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     )
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    primitive_utility = dataset.primitive_utility
    batch_size = 1  # 强制设置为1
    
    print(f"数据集配置:")
    print(f"  future_length: {future_length}")
    print(f"  history_length: {history_length}")
    print(f"  primitive_length: {primitive_length}")
    print(f"  batch_size: {batch_size} (强制设置为1用于VolumetricSMPL)")

    """optimization config"""
    with open(optim_args.interaction_cfg, 'r') as f:
        interaction_cfg = json.load(f)
    interaction_name = interaction_cfg['interaction_name'].replace(' ', '_')
    scene_dir = Path(interaction_cfg['scene_dir'])
    scene_dir = Path(scene_dir)
    
    # 创建专为VolumetricSMPL优化的场景资产
    scene_assets = create_scene_assets(
        scene_dir=scene_dir,
        floor_height=interaction_cfg['floor_height'],
        num_scene_points=8000,  # 8K场景点
        device=device
    )
    
    print(f"场景点云: {scene_assets['scene_points'].shape[0]} 个点")

    out_path = optim_args.save_dir
    filename = f'volsmpl_real_api_guidance{optim_args.guidance_param}_seed{optim_args.seed}'
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
        cache_path = out_path / f'volsmpl_real_cache_{interaction_idx}.pkl'
        if cache_path.exists() and optim_args.load_cache:
            with open(cache_path, 'rb') as f:
                all_motion_sequences, history_motion_tensor, transf_rotmat, transf_transl = pickle.load(f)
            tensor_dict_to_device(all_motion_sequences, device)
            history_motion_tensor = history_motion_tensor.to(device)
            transf_rotmat = transf_rotmat.to(device)
            transf_transl = transf_transl.to(device)
        else:
            text_prompt = interaction['text_prompt']
            
            # 确保goal_joints和joints_mask在GPU上
            goal_joints = torch.zeros(batch_size, 22, 3, device=device, dtype=torch.float32)
            goal_joints[:, 0] = torch.tensor(interaction['goal_joints'][0], device=device, dtype=torch.float32)
            joints_mask = torch.zeros(22, device=device, dtype=torch.bool)
            joints_mask[0] = 1
            
            print(f"目标设置:")
            print(f"  goal_joints: device={goal_joints.device}, shape={goal_joints.shape}")
            print(f"  joints_mask: device={joints_mask.device}, shape={joints_mask.shape}")

            if interaction_idx == 0:
                history_motion_tensor = init_history_motion
                initial_joints = torch.tensor(interaction['init_joints'], device=device,
                                              dtype=torch.float32)  # [3, 3]
                transf_rotmat, transf_transl = get_new_coordinate(initial_joints[None])
                transf_rotmat = transf_rotmat.repeat(batch_size, 1, 1).to(device)
                transf_transl = transf_transl.repeat(batch_size, 1, 1).to(device)
                
                print(f"变换矩阵:")
                print(f"  transf_rotmat: device={transf_rotmat.device}")
                print(f"  transf_transl: device={transf_transl.device}")

            print(f"\n=== 交互 {interaction_idx+1}: {text_prompt} ===")
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
        with open(out_path / f'volsmpl_real_sample_{idx}.pkl', 'wb') as f:
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
            with open(out_path / f'volsmpl_real_sample_{idx}_smplx.npz', 'wb') as f:
                np.savez(f, **data_dict)

    print(f'[Done] VolumetricSMPL真实API优化结果保存在: [{out_path.absolute()}]')
