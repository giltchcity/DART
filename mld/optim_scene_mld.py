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
    weight_geometric_collision: float = 0.5  # 新增：几何碰撞权重
    load_cache: int = 0
    contact_thresh: float = 0.03
    init_noise_scale: float = 1.0
    
    # 改进的VolumetricSMPL参数
    volsmpl_max_frames: int = 10  # 增加到10帧
    volsmpl_model_folder: str = "./data/smplx_lockedhead_20230207/models_lockedhead"
    scene_points_density: int = 16000  # 增加到16K点
    adaptive_frame_selection: int = 1  # 智能帧选择
    multi_resolution_sdf: int = 1  # 多分辨率SDF查询
    
    interaction_cfg: str = './data/optim_interaction/climb_up_stairs.json'


def calc_jerk(joints):
    vel = joints[:, 1:] - joints[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    jerk = acc[:, 1:] - acc[:, :-1]
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))
    jerk = jerk.amax(dim=[1, 2])
    return jerk.mean()

def create_enhanced_scene_point_cloud(scene_mesh, num_points=16000, ensure_z_up=True, floor_height=0.0):
    """
    创建增强的场景点云，重点采样关键区域
    """
    mesh = scene_mesh.copy()
    
    if ensure_z_up:
        bounds = mesh.bounds
        min_z = bounds[0][2]
        if abs(min_z - floor_height) > 1e-5:
            translation = np.array([0, 0, floor_height - min_z])
            mesh.apply_translation(translation)
    
    # 分层采样策略
    base_points = int(num_points * 0.6)  # 60%基础采样
    dense_points = int(num_points * 0.4)  # 40%密集采样重要区域
    
    # 基础均匀采样
    points_base, face_indices_base = mesh.sample(base_points, return_index=True)
    
    # 识别关键区域（椅子、楼梯等）并密集采样
    bounds = mesh.bounds
    y_range = bounds[1][1] - bounds[0][1]
    z_range = bounds[1][2] - bounds[0][2]
    
    # 定义感兴趣区域（椅子高度、楼梯区域等）
    interest_regions = [
        # 椅子区域 (坐姿高度)
        {'y_min': bounds[0][1] + y_range * 0.3, 'y_max': bounds[1][1] - y_range * 0.1,
         'z_min': 0.3, 'z_max': 1.2},
        # 楼梯区域 (步行高度)
        {'y_min': bounds[0][1], 'y_max': bounds[0][1] + y_range * 0.4,
         'z_min': 0.0, 'z_max': 0.5},
    ]
    
    # 在感兴趣区域密集采样
    dense_points_collected = []
    for region in interest_regions:
        try:
            # 创建区域mask（这是一个简化版本，实际实现可能需要更复杂的几何操作）
            region_samples, region_faces = mesh.sample(dense_points // len(interest_regions))
            region_mask = (
                (region_samples[:, 1] >= region['y_min']) & 
                (region_samples[:, 1] <= region['y_max']) &
                (region_samples[:, 2] >= region['z_min']) & 
                (region_samples[:, 2] <= region['z_max'])
            )
            if region_mask.sum() > 0:
                dense_points_collected.append(region_samples[region_mask])
        except:
            continue
    
    # 合并所有点
    if dense_points_collected:
        points_dense = np.vstack(dense_points_collected)
        points_all = np.vstack([points_base, points_dense])
    else:
        # 如果区域采样失败，回退到均匀采样
        points_all, face_indices_base = mesh.sample(num_points, return_index=True)
    
    # 如果点数超过目标，随机选择
    if points_all.shape[0] > num_points:
        indices = np.random.choice(points_all.shape[0], num_points, replace=False)
        points_all = points_all[indices]
    
    # 计算法向量（使用最近邻近似）
    try:
        face_normals = mesh.face_normals[face_indices_base[:len(points_all)]]
    except:
        face_normals = np.repeat([[0, 0, 1]], len(points_all), axis=0)
    
    points_tensor = torch.tensor(points_all, dtype=torch.float32)
    normals_tensor = torch.tensor(face_normals, dtype=torch.float32)
    
    metadata = {
        'num_points': len(points_all),
        'bounds': mesh.bounds.tolist(),
        'center': mesh.centroid.tolist(),
        'floor_height': floor_height,
        'has_normals': True,
        'enhanced_sampling': True
    }
    
    print(f"🔍 增强场景点云: {len(points_all)}点 (目标{num_points})")
    
    return points_tensor, normals_tensor, metadata

def create_scene_assets(scene_dir, floor_height=0.0, num_scene_points=16000, device="cuda"):
    """创建增强的场景资产"""
    scene_with_floor_mesh = trimesh.load(scene_dir / 'scene_with_floor.obj', process=False, force='mesh')
    
    # 使用增强的点云生成
    scene_points, scene_normals, scene_metadata = create_enhanced_scene_point_cloud(
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

class EnhancedVolumetricSMPLCollisionCalculator:
    """
    增强版VolumetricSMPL碰撞检测器 - 多重验证机制
    """
    def __init__(self, model_folder, device="cuda", enable_geometric_fallback=True):
        self.model_folder = model_folder
        self.device = device
        self.smpl_model_cache = {}
        self.enable_geometric_fallback = enable_geometric_fallback
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
    
    def debug_volumetric_smpl_failure(self, model, smpl_output, scene_points, frame_idx=97):
        """
        深度调试VolumetricSMPL为什么检测不到明显穿透
        """
        print(f"\n🔬 VolumetricSMPL失效调试 (帧{frame_idx}):")
        
        # 1. 验证SMPL mesh的实际范围
        vertices = smpl_output.vertices[0]  # [V, 3]
        print(f"SMPL mesh范围:")
        print(f"  X: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
        print(f"  Y: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]") 
        print(f"  Z: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
        
        # 2. 检查椅子区域的场景点
        chair_mask = (
            (scene_points[:, 1] > 2.4) & (scene_points[:, 1] < 2.5) &  # Y: 椅子区域
            (scene_points[:, 2] > 0.5) & (scene_points[:, 2] < 0.7)    # Z: 坐垫高度
        )
        chair_points = scene_points[chair_mask]
        print(f"\n椅子区域分析:")
        print(f"  椅子点数: {chair_points.shape[0]}")
        if chair_points.shape[0] > 0:
            print(f"  椅子范围: Y[{chair_points[:, 1].min():.3f}, {chair_points[:, 1].max():.3f}] Z[{chair_points[:, 2].min():.3f}, {chair_points[:, 2].max():.3f}]")
        
        # 3. 手动测试椅子内部明显的点
        if chair_points.shape[0] > 0:
            # 选择椅子中心的一些点进行测试
            test_points = chair_points[:min(50, chair_points.shape[0])]
            test_points_batch = test_points.unsqueeze(0)
            
            print(f"\n手动SDF测试 ({test_points.shape[0]}个椅子内部点):")
            try:
                sdf_values = model.volume.query(test_points_batch, smpl_output)
                sdf_values = sdf_values.squeeze(0)
                
                negative_count = (sdf_values < 0).sum().item()
                near_zero_count = (torch.abs(sdf_values) < 0.01).sum().item()
                
                print(f"  SDF值范围: [{sdf_values.min():.6f}, {sdf_values.max():.6f}]")
                print(f"  负值数量: {negative_count}/{test_points.shape[0]} (应该>0如果有穿透)")
                print(f"  接近0: {near_zero_count}/{test_points.shape[0]}")
                
                if negative_count == 0:
                    print(f"  🚨 问题确认: 椅子内部点的SDF都是正值!")
                    print(f"  🚨 这说明VolumetricSMPL认为椅子不在人体内部!")
                    
                    # 显示几个具体例子
                    for i in range(min(5, test_points.shape[0])):
                        point = test_points[i]
                        sdf = sdf_values[i]
                        print(f"    点{i}: ({point[0]:.3f},{point[1]:.3f},{point[2]:.3f}) SDF={sdf:.6f}")
                
            except Exception as e:
                print(f"  ❌ SDF查询失败: {e}")
        
        # 4. 验证人体中心是否与椅子重叠
        pelvis_pos = vertices.mean(dim=0)  # 人体中心近似
        print(f"\n重叠分析:")
        print(f"  人体中心(近似): ({pelvis_pos[0]:.3f}, {pelvis_pos[1]:.3f}, {pelvis_pos[2]:.3f})")
        
        if chair_points.shape[0] > 0:
            chair_center = chair_points.mean(dim=0)
            print(f"  椅子中心: ({chair_center[0]:.3f}, {chair_center[1]:.3f}, {chair_center[2]:.3f})")
            
            # 计算人体包围盒与椅子的重叠
            human_bbox_min = vertices.min(dim=0)[0]
            human_bbox_max = vertices.max(dim=0)[0]
            chair_bbox_min = chair_points.min(dim=0)[0]
            chair_bbox_max = chair_points.max(dim=0)[0]
            
            overlap_x = max(0, min(human_bbox_max[0], chair_bbox_max[0]) - max(human_bbox_min[0], chair_bbox_min[0]))
            overlap_y = max(0, min(human_bbox_max[1], chair_bbox_max[1]) - max(human_bbox_min[1], chair_bbox_min[1]))
            overlap_z = max(0, min(human_bbox_max[2], chair_bbox_max[2]) - max(human_bbox_min[2], chair_bbox_min[2]))
            
            print(f"  包围盒重叠: X={overlap_x:.3f} Y={overlap_y:.3f} Z={overlap_z:.3f}")
            
            if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
                print(f"  ✅ 包围盒有重叠 - 应该检测到穿透!")
            else:
                print(f"  ❌ 包围盒无重叠 - 可能是坐标系问题")
        
        # 5. 测试VolumetricSMPL的query方法是否正常工作
        print(f"\n🔧 VolumetricSMPL功能测试:")
        try:
            # 测试一些明显在人体外部的点
            far_points = torch.tensor([
                [-5.0, 0.0, 1.0],  # 远离人体的点
                [5.0, 0.0, 1.0],
            ], device=scene_points.device).unsqueeze(0)
            
            far_sdf = model.volume.query(far_points, smpl_output)
            print(f"  远离点SDF: {far_sdf.squeeze(0).tolist()} (应该是大正值)")
            
            # 测试一些接近人体表面的点  
            surface_points = vertices[:5].unsqueeze(0)  # 取人体表面的几个点
            surface_sdf = model.volume.query(surface_points, smpl_output) 
            print(f"  表面点SDF: {surface_sdf.squeeze(0)[:3].tolist()} (应该接近0)")
            
            # 如果上面都正常，说明VolumetricSMPL本身工作正常
            # 问题可能在于椅子点的位置或者坐标系
            
        except Exception as e:
            print(f"  ❌ VolumetricSMPL基础功能异常: {e}")
        
        print(f"="*60)
    
    def select_adaptive_frames(self, motion_sequences, max_frames, task_type="general"):
        """
        智能帧选择：根据任务类型和动作特征选择关键帧
        """
        B, T, _, _ = motion_sequences['joints'].shape
        
        if task_type == "sit":
            # 坐下任务：重点关注最后的坐下阶段
            sit_start = max(0, T - max_frames)
            frame_indices = torch.arange(sit_start, T, device=self.device)
        elif task_type == "climb":
            # 爬楼梯：均匀分布，重点关注中间运动阶段
            frame_indices = torch.linspace(T//4, T-1, max_frames, dtype=torch.long, device=self.device)
        else:
            # 通用：均匀分布 + 最后几帧
            uniform_frames = torch.linspace(0, T-1, max_frames-2, dtype=torch.long, device=self.device)
            final_frames = torch.tensor([T-2, T-1], dtype=torch.long, device=self.device)
            frame_indices = torch.cat([uniform_frames, final_frames])
        
        return frame_indices.clamp(0, T-1)
    
    def geometric_collision_check(self, smpl_vertices, scene_points, threshold=0.05):
        """
        几何碰撞检测：作为VolumetricSMPL的补充验证
        """
        # 计算SMPL顶点到场景点的最小距离
        distances = torch.cdist(smpl_vertices, scene_points)  # [V, S]
        min_distances = distances.min(dim=1)[0]  # [V]
        
        # 穿透检测：如果顶点距离场景点过近
        penetration_count = (min_distances < threshold).sum().item()
        penetration_ratio = penetration_count / smpl_vertices.shape[0]
        
        # 碰撞损失：距离越近损失越大
        collision_loss = torch.relu(threshold - min_distances).mean()
        
        return {
            'penetration_count': penetration_count,
            'penetration_ratio': penetration_ratio,
            'collision_loss': collision_loss,
            'min_distance': min_distances.min().item(),
            'avg_distance': min_distances.mean().item()
        }
    
    def analyze_interaction_quality(self, motion_sequences, scene_points, frame_indices, task_type="general"):
        """
        深度分析人体-场景交互质量
        """
        joints = motion_sequences['joints']
        B, T, J, _ = joints.shape
        
        scene_min = scene_points.min(dim=0)[0]
        scene_max = scene_points.max(dim=0)[0]
        scene_center = scene_points.mean(dim=0)
        
        print(f"\n🔬 深度交互分析 ({task_type}):")
        print(f"场景范围: X[{scene_min[0]:.2f}, {scene_max[0]:.2f}] Y[{scene_min[1]:.2f}, {scene_max[1]:.2f}] Z[{scene_min[2]:.2f}, {scene_max[2]:.2f}]")
        
        interaction_stats = {
            'close_contact_frames': 0,
            'penetration_frames': 0,
            'min_distances': [],
            'geometric_collisions': [],
            'critical_frames': []
        }
        
        for i, frame_idx in enumerate(frame_indices):
            frame_joints = joints[0, frame_idx]
            pelvis_pos = frame_joints[0]
            
            # 计算所有关节到场景的距离
            distances = torch.cdist(frame_joints.unsqueeze(0), scene_points.unsqueeze(0))[0]
            min_distance = distances.min().item()
            closest_joint_idx = distances.min(dim=1)[0].argmin().item()
            
            interaction_stats['min_distances'].append(min_distance)
            
            # 判断交互状态
            contact_status = "🟢自由"
            if min_distance < 0.08:
                contact_status = "🟡接近"
                interaction_stats['close_contact_frames'] += 1
            if min_distance < 0.03:
                contact_status = "🔴接触"
            if min_distance < 0.01:
                contact_status = "❌穿透"
                interaction_stats['penetration_frames'] += 1
                interaction_stats['critical_frames'].append(frame_idx)
            
            # 特别关注坐下任务的最后几帧
            is_critical = (task_type == "sit" and i >= len(frame_indices) - 3)
            
            print(f"  帧{frame_idx:2d}: 骨盆({pelvis_pos[0]:.2f},{pelvis_pos[1]:.2f},{pelvis_pos[2]:.2f}) "
                  f"距离={min_distance:.3f}m 关节{closest_joint_idx} {contact_status} "
                  f"{'⭐关键帧' if is_critical else ''}")
        
        print(f"交互统计: 接近={interaction_stats['close_contact_frames']}/{len(frame_indices)} "
              f"穿透={interaction_stats['penetration_frames']}/{len(frame_indices)} "
              f"关键帧={len(interaction_stats['critical_frames'])}")
        
        return interaction_stats
    
    def multi_resolution_sdf_query(self, model, scene_points, smpl_output, resolution_levels=[1.0, 0.5, 0.25]):
        """
        多分辨率SDF查询：在不同密度下查询以提高检测精度
        """
        total_penetration = 0
        total_contact = 0
        total_points = 0
        
        for resolution in resolution_levels:
            # 按分辨率采样点
            num_points = int(scene_points.shape[0] * resolution)
            if num_points < 100:
                continue
                
            indices = torch.randperm(scene_points.shape[0])[:num_points]
            sample_points = scene_points[indices].unsqueeze(0)
            
            try:
                sdf_values = model.volume.query(sample_points, smpl_output)
                
                penetration = (sdf_values < -0.01).sum().item()
                contact = ((sdf_values >= -0.01) & (sdf_values <= 0.01)).sum().item()
                
                total_penetration += penetration
                total_contact += contact
                total_points += num_points
                
                print(f"    分辨率{resolution:.2f}: {num_points}点 -> 穿透{penetration} 接触{contact}")
                
            except Exception as e:
                print(f"    分辨率{resolution:.2f}查询失败: {e}")
                continue
        
        return {
            'total_penetration': total_penetration,
            'total_contact': total_contact,
            'total_points': total_points,
            'penetration_ratio': total_penetration / max(total_points, 1)
        }
    
    def enhanced_collision_detection(self, motion_sequences, scene_points, frame_indices, task_type="general"):
        """
        增强版碰撞检测：VolumetricSMPL + 几何验证 + 多重检查
        """
        gender = motion_sequences['gender']
        model = self.get_smpl_model(gender)
        
        B, T, _, _ = motion_sequences['joints'].shape
        num_scene_points = scene_points.shape[0]
        
        scene_points = scene_points.to(self.device)
        
        # 深度交互分析
        interaction_stats = self.analyze_interaction_quality(motion_sequences, scene_points, frame_indices, task_type)
        
        # 初始化结果
        sdf_tensor = torch.zeros(len(frame_indices), num_scene_points, device=self.device)
        total_volsmpl_collision = torch.tensor(0.0, device=self.device)
        total_volsmpl_self_collision = torch.tensor(0.0, device=self.device)
        total_geometric_collision = torch.tensor(0.0, device=self.device)
        
        sdf_summary = {'penetration_points': 0, 'contact_points': 0, 'free_points': 0}
        geometric_summary = {'avg_penetration_ratio': 0, 'total_geometric_loss': 0}
        
        for i, frame_idx in enumerate(frame_indices):
            batch_idx = 0
            
            # SMPL参数提取和转换
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
            
            # === 1. VolumetricSMPL检测 ===
            scene_points_batch = scene_points.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            
            volsmpl_success = False
            try:
                # 多分辨率SDF查询
                multi_res_result = self.multi_resolution_sdf_query(model, scene_points, smpl_output)
                
                # 标准SDF查询
                sdf_values = model.volume.query(scene_points_batch, smpl_output)
                sdf_tensor[i] = sdf_values.squeeze(0).to(self.device)
                
                penetration = (sdf_values < -0.01).sum().item()
                contact = ((sdf_values >= -0.01) & (sdf_values <= 0.01)).sum().item()
                free = (sdf_values > 0.01).sum().item()
                
                sdf_summary['penetration_points'] += penetration
                sdf_summary['contact_points'] += contact
                sdf_summary['free_points'] += free
                
                volsmpl_success = True
                
                if i == 0 or frame_idx in interaction_stats.get('critical_frames', []):
                    print(f"  🔍 VolumetricSMPL 帧{frame_idx}: 穿透{penetration} 接触{contact} 自由{free}")
                    print(f"    SDF范围[{sdf_values.min():.3f}, {sdf_values.max():.3f}] 多分辨率穿透率{multi_res_result['penetration_ratio']:.3f}")
                
            except Exception as e:
                print(f"  ❌ VolumetricSMPL失败 帧{frame_idx}: {e}")
                sdf_tensor[i] = torch.zeros(num_scene_points, device=self.device)
            
            # === 2. 几何碰撞检测 (备用验证) ===
            if self.enable_geometric_fallback:
                geometric_result = self.geometric_collision_check(
                    smpl_output.vertices[0], scene_points, threshold=0.05
                )
                
                total_geometric_collision += geometric_result['collision_loss']
                geometric_summary['avg_penetration_ratio'] += geometric_result['penetration_ratio']
                geometric_summary['total_geometric_loss'] += geometric_result['collision_loss'].item()
                
                # 如果VolumetricSMPL失败但几何检测发现问题，发出警告
                if not volsmpl_success and geometric_result['penetration_count'] > 0:
                    print(f"  ⚠️  几何检测发现穿透但VolumetricSMPL失败! 穿透顶点{geometric_result['penetration_count']}")
                
                if i == 0 or frame_idx in interaction_stats.get('critical_frames', []):
                    print(f"  🔧 几何验证 帧{frame_idx}: 穿透顶点{geometric_result['penetration_count']} "
                          f"最小距离{geometric_result['min_distance']:.3f}m")
            
            # === 3. VolumetricSMPL损失计算 ===
            try:
                if not hasattr(smpl_output, 'full_pose'):
                    smpl_output.full_pose = full_pose
                
                # 场景碰撞
                scene_collision_loss = model.volume.collision_loss(scene_points_batch, smpl_output)
                if isinstance(scene_collision_loss, (list, tuple)):
                    scene_collision_loss = scene_collision_loss[0]
                if scene_collision_loss.dim() == 0:
                    scene_collision_loss = scene_collision_loss.unsqueeze(0)
                scene_collision_loss = scene_collision_loss.to(self.device)
                total_volsmpl_collision += scene_collision_loss.mean()
                
                # 自碰撞
                self_collision_loss = model.volume.self_collision_loss(smpl_output)
                if isinstance(self_collision_loss, (list, tuple)):
                    self_collision_loss = self_collision_loss[0]
                if self_collision_loss.dim() == 0:
                    self_collision_loss = self_collision_loss.unsqueeze(0)
                elif self_collision_loss.dim() > 1:
                    self_collision_loss = self_collision_loss.mean()
                self_collision_loss = self_collision_loss.to(self.device)
                total_volsmpl_self_collision += self_collision_loss.mean()
                
            except Exception as e:
                if i == 0:
                    print(f"  ❌ VolumetricSMPL损失计算失败: {e}")
                scene_collision_loss = torch.tensor(0.0, device=self.device)
                self_collision_loss = torch.tensor(0.0, device=self.device)
                total_volsmpl_collision += scene_collision_loss
                total_volsmpl_self_collision += self_collision_loss
        
        # 最终统计
        num_frames = len(frame_indices)
        num_total_points = num_frames * num_scene_points
        
        avg_volsmpl_collision = (total_volsmpl_collision.mean() / num_frames).squeeze()
        avg_volsmpl_self_collision = (total_volsmpl_self_collision.mean() / num_frames).squeeze()
        avg_geometric_collision = (total_geometric_collision.mean() / num_frames).squeeze()
        
        # 检测不一致性
        volsmpl_penetration_ratio = sdf_summary['penetration_points'] / max(num_total_points, 1)
        geometric_penetration_ratio = geometric_summary['avg_penetration_ratio'] / max(num_frames, 1)
        
        print(f"\n📊 增强检测总结:")
        print(f"  VolumetricSMPL: 穿透{sdf_summary['penetration_points']}/{num_total_points}点 ({volsmpl_penetration_ratio:.3f})")
        print(f"  几何验证: 平均穿透率{geometric_penetration_ratio:.3f}")
        print(f"  💥 损失: VolumetricSMPL场景={avg_volsmpl_collision.item():.6f} 自碰撞={avg_volsmpl_self_collision.item():.6f}")
        print(f"         几何补充={avg_geometric_collision.item():.6f}")
        
        # 🎯 关键添加：不一致性检测和深度调试
        inconsistency_threshold = 0.02  # 如果几何检测穿透率>2%但VolumetricSMPL<0.1%，则认为不一致
        if volsmpl_penetration_ratio < 0.001 and geometric_penetration_ratio > inconsistency_threshold:
            print(f"  🚨 检测不一致警告: VolumetricSMPL穿透率{volsmpl_penetration_ratio:.3f} vs 几何穿透率{geometric_penetration_ratio:.3f}!")
            print(f"  🔬 启动深度调试分析...")
            
            # 取最后一帧（最可能有穿透的帧）进行深度调试
            last_frame_idx = frame_indices[-1].item()
            
            # 重新计算最后一帧的SMPL输出用于调试
            batch_idx = 0
            global_orient_matrix = motion_sequences['global_orient'][batch_idx, last_frame_idx].detach().clone().to(device=self.device, dtype=torch.float32)
            body_pose_matrix = motion_sequences['body_pose'][batch_idx, last_frame_idx].detach().clone().to(device=self.device, dtype=torch.float32) 
            transl = motion_sequences['transl'][batch_idx, last_frame_idx].detach().clone().to(device=self.device, dtype=torch.float32)
            betas = motion_sequences['betas'][batch_idx, last_frame_idx, :10].detach().clone().to(device=self.device, dtype=torch.float32)
            
            global_orient_aa = transforms.matrix_to_axis_angle(global_orient_matrix).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            body_pose_aa = transforms.matrix_to_axis_angle(body_pose_matrix).to(device=self.device, dtype=torch.float32)
            body_pose_aa_flat = body_pose_aa.reshape(1, -1).to(device=self.device, dtype=torch.float32)
            transl_batch = transl.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            betas_batch = betas.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            
            debug_smpl_output = model(
                betas=betas_batch,
                global_orient=global_orient_aa,
                body_pose=body_pose_aa_flat,
                transl=transl_batch,
                return_verts=True,
                pose2rot=True
            )
            
            full_pose = torch.cat([global_orient_aa, body_pose_aa_flat], dim=1).to(device=self.device, dtype=torch.float32)
            debug_smpl_output.full_pose = full_pose
            
            # 🎯 在这里调用深度调试
            self.debug_volumetric_smpl_failure(model, debug_smpl_output, scene_points, last_frame_idx)
        
        return sdf_tensor, avg_volsmpl_collision, avg_volsmpl_self_collision, avg_geometric_collision
    
    def calculate_enhanced_contact_loss(self, motion_sequences, scene_points, frame_indices, contact_thresh=0.03):
        """增强的接触损失计算"""
        try:
            scene_points = scene_points.to(self.device)
            sdf_tensor, _, _, _ = self.enhanced_collision_detection(motion_sequences, scene_points, frame_indices)
            
            # 基于SDF的接触损失
            contact_loss = torch.relu(contact_thresh - torch.abs(sdf_tensor)).mean()
            contact_loss = contact_loss.to(self.device)
            
            return contact_loss
        except Exception as e:
            print(f"❌ 增强接触损失计算失败: {e}")
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
        
    # 创建增强版VolumetricSMPL计算器
    enhanced_calculator = EnhancedVolumetricSMPLCollisionCalculator(
        model_folder=optim_args.volsmpl_model_folder,
        device=device,
        enable_geometric_fallback=True
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
    
    # 判断任务类型
    task_type = "sit" if "sit" in text_prompt.lower() else ("climb" if "climb" in text_prompt.lower() else "general")
    
    print(f"\n🚀 开始增强VolumetricSMPL优化 (任务类型: {task_type})")
    print(f"优化: {optim_steps}步, 学习率{lr}, 帧数{optim_args.volsmpl_max_frames}")
    print("="*80)
    
    for i in tqdm(range(optim_steps), desc="增强优化进度"):
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
        
        # 智能帧选择
        if optim_args.adaptive_frame_selection:
            frame_indices = enhanced_calculator.select_adaptive_frames(
                motion_sequences, optim_args.volsmpl_max_frames, task_type
            )
        else:
            max_frames = min(optim_args.volsmpl_max_frames, T)
            frame_indices = torch.linspace(0, T-1, max_frames, dtype=torch.long)
        
        scene_points = scene_assets['scene_points'].to(device)
        
        try:
            if i % 20 == 0:  # 每20步详细分析
                print(f"\n🔍 第{i}步增强分析 ({task_type}):")
            
            # 使用增强检测系统
            sdf_tensor, volsmpl_collision, volsmpl_self_collision, geometric_collision = \
                enhanced_calculator.enhanced_collision_detection(
                    motion_sequences, scene_points, frame_indices, task_type
                )
            
            # 增强接触损失
            contact_loss = enhanced_calculator.calculate_enhanced_contact_loss(
                motion_sequences, scene_points, frame_indices, optim_args.contact_thresh
            )
            
            # 确保损失在正确设备上并为标量
            volsmpl_collision = volsmpl_collision.to(device)
            volsmpl_self_collision = volsmpl_self_collision.to(device)
            geometric_collision = geometric_collision.to(device)
            contact_loss = contact_loss.to(device)
            
            for loss in [volsmpl_collision, volsmpl_self_collision, geometric_collision, contact_loss]:
                if loss.dim() > 0:
                    loss = loss.mean()
            
        except Exception as e:
            if i % 20 == 0:
                print(f"❌ 增强VolumetricSMPL计算失败: {e}")
                import traceback
                traceback.print_exc()
            
            volsmpl_collision = torch.tensor(0.0, device=device)
            volsmpl_self_collision = torch.tensor(0.0, device=device)
            geometric_collision = torch.tensor(0.0, device=device)
            contact_loss = torch.tensor(0.0, device=device)
        
        # 其他损失项
        goal_joints_gpu = goal_joints.to(device)
        joints_mask_gpu = joints_mask.to(device)
        
        loss_joints = criterion(
            motion_sequences['joints'][:, -1, joints_mask_gpu], 
            goal_joints_gpu[:, joints_mask_gpu]
        )
        loss_jerk = calc_jerk(motion_sequences['joints'])
        
        loss_joints = loss_joints.to(device)
        loss_jerk = loss_jerk.to(device)
        
        # 增强的总损失计算
        total_loss = (
            loss_joints +  # 目标损失
            optim_args.weight_collision * volsmpl_collision +  # VolumetricSMPL场景碰撞
            optim_args.weight_self_collision * volsmpl_self_collision +  # VolumetricSMPL自碰撞
            optim_args.weight_geometric_collision * geometric_collision +  # 几何碰撞补充
            optim_args.weight_contact * contact_loss +  # 接触损失
            optim_args.weight_jerk * loss_jerk  # 平滑性
        )
        
        total_loss = total_loss.to(device)

        total_loss.backward()
        if optim_args.optim_unit_grad:
            noise.grad.data /= noise.grad.norm(p=2, dim=reduction_dims, keepdim=True).clamp(min=1e-6)
        optimizer.step()
        
        # 详细进度报告
        if i % 20 == 0:
            print(f'📊 [{i:3d}/{optim_steps}] 总损失={total_loss.item():.4f}')
            print(f'  目标={loss_joints.item():.4f} | VolumetricSMPL: 场景={volsmpl_collision.item():.4f} 自碰撞={volsmpl_self_collision.item():.4f}')
            print(f'  几何补充={geometric_collision.item():.4f} | 接触={contact_loss.item():.4f} | 平滑={loss_jerk.item():.4f}')

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

    print("🚀 初始化增强VolumetricSMPL系统")
    print(f"设备: {device} | 优化步数: {optim_args.optim_steps}")
    print(f"场景点密度: {optim_args.scene_points_density} | 检测帧数: {optim_args.volsmpl_max_frames}")
    print(f"权重: VolumetricSMPL(场景={optim_args.weight_collision}, 自碰撞={optim_args.weight_self_collision}) "
          f"几何补充={optim_args.weight_geometric_collision} 接触={optim_args.weight_contact}")

    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(optim_args.denoiser_checkpoint, device)
    denoiser_checkpoint = Path(optim_args.denoiser_checkpoint)
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / 'enhanced_volsmpl_optim'
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
    
    # 创建增强场景资产
    scene_assets = create_scene_assets(
        scene_dir=scene_dir,
        floor_height=interaction_cfg['floor_height'],
        num_scene_points=optim_args.scene_points_density,
        device=device
    )
    
    print(f"🏢 增强场景加载完成: {scene_assets['scene_points'].shape[0]}点")

    out_path = optim_args.save_dir
    filename = f'enhanced_volsmpl_guidance{optim_args.guidance_param}_seed{optim_args.seed}'
    if optim_args.respacing != '':
        filename = f'{optim_args.respacing}_{filename}'
    if optim_args.zero_noise:
        filename = f'zero_noise_{filename}'
    if optim_args.use_predicted_joints:
        filename = f'use_pred_joints_{filename}'
    filename = f'{interaction_name}_{filename}'
    filename = f'{filename}_vol{optim_args.weight_collision}_self{optim_args.weight_self_collision}_geo{optim_args.weight_geometric_collision}_contact{optim_args.weight_contact}_jerk{optim_args.weight_jerk}_frames{optim_args.volsmpl_max_frames}_points{optim_args.scene_points_density}'
    
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
        cache_path = out_path / f'enhanced_volsmpl_cache_{interaction_idx}.pkl'
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

            print(f"\n🎯 开始增强交互 {interaction_idx+1}: {text_prompt}")
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
        with open(out_path / f'enhanced_volsmpl_sample_{idx}.pkl', 'wb') as f:
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
            with open(out_path / f'enhanced_volsmpl_sample_{idx}_smplx.npz', 'wb') as f:
                np.savez(f, **data_dict)

    print(f'✅ 增强分析完成! 结果保存在: {out_path.absolute()}')
