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

# Define hand joint indices for SMPL model
LEFT_HAND_IDX = 20  # Left wrist in SMPL
RIGHT_HAND_IDX = 21  # Right wrist in SMPL

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
    batch_size: int = 2  # Changed to 2 for two people

    optim_lr: float = 0.01
    optim_steps: int = 300
    optim_unit_grad: int = 1
    optim_anneal_lr: int = 1
    weight_jerk: float = 0.0
    weight_collision: float = 0.0
    weight_contact: float = 0.0
    weight_skate: float = 0.0
    weight_interaction: float = 1.0  # New weight for human-human interaction
    load_cache: int = 0
    contact_thresh: float = 0.03
    init_noise_scale: float = 1.0

    interaction_cfg: str = './data/optim_interaction/high_five.json'


import torch.nn.functional as F
def calc_point_sdf(scene_assets, points):
    device = points.device
    scene_sdf_config = scene_assets['scene_sdf_config']
    scene_sdf_grid = scene_assets['scene_sdf_grid']
    sdf_size = scene_sdf_config['size']
    sdf_scale = scene_sdf_config['scale']
    sdf_scale = torch.tensor(sdf_scale, dtype=torch.float32, device=device).reshape(1, 1, 1)
    sdf_center = scene_sdf_config['center']
    sdf_center = torch.tensor(sdf_center, dtype=torch.float32, device=device).reshape(1, 1, 3)
    batch_size, num_points, _ = points.shape
    points = (points - sdf_center) * sdf_scale
    sdf_values = F.grid_sample(scene_sdf_grid.unsqueeze(0),
                               points[:, :, [2, 1, 0]].view(batch_size, num_points, 1, 1, 3),
                               padding_mode='border',
                               align_corners=True
                               ).reshape(batch_size, num_points)
    sdf_values = sdf_values / sdf_scale.squeeze(-1)
    return sdf_values

def calc_jerk(joints):
    vel = joints[:, 1:] - joints[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    jerk = acc[:, 1:] - acc[:, :-1]
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))
    jerk = jerk.amax(dim=[1, 2])
    return jerk.mean()

def calc_interaction_loss(joints_person1, joints_person2, interaction_frame, interaction_type='high_five'):
    """Calculate loss for human-human interaction"""
    if interaction_type == 'high_five':
        # Get hand positions at interaction frame
        left_hand_p1 = joints_person1[:, interaction_frame, LEFT_HAND_IDX]
        right_hand_p1 = joints_person1[:, interaction_frame, RIGHT_HAND_IDX]
        left_hand_p2 = joints_person2[:, interaction_frame, LEFT_HAND_IDX]
        right_hand_p2 = joints_person2[:, interaction_frame, RIGHT_HAND_IDX]
        
        # Calculate distance between hands (person1's right hand with person2's left hand)
        hand_distance = torch.norm(right_hand_p1 - left_hand_p2, dim=-1)
        
        # Loss encourages hands to be close at interaction frame
        interaction_loss = hand_distance.mean()
        
        # Also ensure hands are at reasonable height (around shoulder level)
        target_height = 1.4  # meters
        height_loss = (right_hand_p1[:, 2] - target_height).abs() + (left_hand_p2[:, 2] - target_height).abs()
        height_loss = height_loss.mean()
        
        return interaction_loss + 0.5 * height_loss
    
    return 0.0

def optimize_two_person(history_motion_tensors, transf_rotmats, transf_transls, text_prompts, 
                       goal_joints_list, joints_masks, interaction_config):
    """Optimize motion for two people with interaction constraints"""
    
    # Process text prompts for both people
    all_texts = []
    all_text_embeddings = []
    num_rollouts = []
    
    for person_idx, text_prompt in enumerate(text_prompts):
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
        
        all_texts.append(texts)
        num_rollouts.append(num_rollout)
        text_embedding = encode_text(dataset.clip_model, texts, force_empty_zero=True).to(dtype=torch.float32, device=device)
        all_text_embeddings.append(text_embedding)
    
    # Ensure both people have same number of rollouts
    assert num_rollouts[0] == num_rollouts[1], "Both people must have same number of rollouts"
    num_rollout = num_rollouts[0]
    
    def rollout_two_person(noises, history_motion_tensors, transf_rotmats, transf_transls):
        motion_sequences_list = [None, None]
        new_history_motions = []
        new_transf_rotmats = []
        new_transf_transls = []
        
        for person_idx in range(2):
            motion_sequences = None
            history_motion = history_motion_tensors[person_idx]
            transf_rotmat = transf_rotmats[person_idx]
            transf_transl = transf_transls[person_idx]
            
            for segment_id in range(num_rollout):
                text_embedding = all_text_embeddings[person_idx][segment_id].expand(1, -1)  # [1, 512]
                guidance_param = torch.ones(1, *denoiser_args.model_args.noise_shape).to(device=device) * optim_args.guidance_param
                y = {
                    'text_embedding': text_embedding,
                    'history_motion_normalized': history_motion,
                    'scale': guidance_param,
                }

                x_start_pred = sample_fn(
                    denoiser_model,
                    (1, *denoiser_args.model_args.noise_shape),
                    clip_denoised=False,
                    model_kwargs={'y': y},
                    skip_timesteps=0,
                    init_image=None,
                    progress=False,
                    noise=noises[person_idx][segment_id],
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
                        'betas': betas[:1, :future_length, :] if segment_id > 0 else betas[:1, :primitive_length, :],
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

                # Update history motion and transforms
                history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames)
                history_feature_dict.update(
                    {
                        'transf_rotmat': transf_rotmat,
                        'transf_transl': transf_transl,
                        'gender': gender,
                        'betas': betas[:1, :history_length, :],
                        'pelvis_delta': pelvis_delta,
                    }
                )
                canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
                    history_feature_dict, use_predicted_joints=optim_args.use_predicted_joints)
                transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
                canonicalized_history_primitive_dict['transf_transl']
                history_motion = primitive_utility.dict_to_tensor(blended_feature_dict)
                history_motion = dataset.normalize(history_motion)
            
            motion_sequences_list[person_idx] = motion_sequences
            new_history_motions.append(history_motion)
            new_transf_rotmats.append(transf_rotmat)
            new_transf_transls.append(transf_transl)
        
        return motion_sequences_list, new_history_motions, new_transf_rotmats, new_transf_transls

    # Initialize optimization
    optim_steps = optim_args.optim_steps
    lr = optim_args.optim_lr
    
    # Create noise for both people
    noises = []
    for person_idx in range(2):
        noise = torch.randn(num_rollout, 1, *denoiser_args.model_args.noise_shape,
                            device=device, dtype=torch.float32)
        noise = noise * optim_args.init_noise_scale
        noise.requires_grad_(True)
        noises.append(noise)
    
    optimizer = torch.optim.Adam(noises, lr=lr)
    criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
    
    # Get interaction frame
    interaction_frame = interaction_config['interaction_frame']
    
    for i in tqdm(range(optim_steps)):
        optimizer.zero_grad()
        if optim_args.optim_anneal_lr:
            frac = 1.0 - i / optim_steps
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow

        motion_sequences_list, new_history_motion_tensors, new_transf_rotmats, new_transf_transls = \
            rollout_two_person(noises, history_motion_tensors, transf_rotmats, transf_transls)
        
        total_loss = 0.0
        
        # Calculate individual losses for each person
        for person_idx in range(2):
            motion_sequences = motion_sequences_list[person_idx]
            global_joints = motion_sequences['joints']  # [1, T, 22, 3]
            
            # Collision loss
            B, T, _, _ = global_joints.shape
            joints_sdf = calc_point_sdf(scene_assets, global_joints.reshape(1, -1, 3)).reshape(B, T, 22)
            negative_sdf_per_frame = (joints_sdf - joint_skin_dist.reshape(1, 1, 22)).clamp(max=0).sum(dim=-1)
            loss_collision = -negative_sdf_per_frame.mean()
            
            # Goal joint loss
            if goal_joints_list[person_idx] is not None:
                loss_joints = criterion(motion_sequences['joints'][:, -1, joints_masks[person_idx]], 
                                      goal_joints_list[person_idx][:, joints_masks[person_idx]])
            else:
                loss_joints = 0.0
            
            # Jerk loss
            loss_jerk = calc_jerk(motion_sequences['joints'])
            
            total_loss += loss_joints + optim_args.weight_collision * loss_collision + optim_args.weight_jerk * loss_jerk
        
        # Interaction loss between two people
        if interaction_frame < motion_sequences_list[0]['joints'].shape[1]:
            loss_interaction = calc_interaction_loss(
                motion_sequences_list[0]['joints'], 
                motion_sequences_list[1]['joints'],
                interaction_frame,
                interaction_config['interaction_type']
            )
            total_loss += optim_args.weight_interaction * loss_interaction
        else:
            loss_interaction = torch.tensor(0.0)
        
        total_loss.backward()
        
        # Gradient clipping
        if optim_args.optim_unit_grad:
            for noise in noises:
                if noise.grad is not None:
                    noise.grad.data /= noise.grad.norm(p=2, dim=list(range(1, len(noise.shape))), keepdim=True).clamp(min=1e-6)
        
        optimizer.step()
        
        if i % 10 == 0:
            print(f'[{i}/{optim_steps}] loss: {total_loss.item():.4f} loss_interaction: {loss_interaction.item():.4f}')
    
    # Detach results
    for person_idx in range(2):
        for key in motion_sequences_list[person_idx]:
            if torch.is_tensor(motion_sequences_list[person_idx][key]):
                motion_sequences_list[person_idx][key] = motion_sequences_list[person_idx][key].detach()
        motion_sequences_list[person_idx]['texts'] = all_texts[person_idx]
    
    return motion_sequences_list, [h.detach() for h in new_history_motion_tensors], \
           [r.detach() for r in new_transf_rotmats], [t.detach() for t in new_transf_transls]

if __name__ == '__main__':
    optim_args = tyro.cli(OptimArgs)
    # Seeding
    random.seed(optim_args.seed)
    np.random.seed(optim_args.seed)
    torch.manual_seed(optim_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = optim_args.torch_deterministic
    device = torch.device(optim_args.device if torch.cuda.is_available() else "cpu")
    optim_args.device = device

    # Load model
    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(optim_args.denoiser_checkpoint, device)
    denoiser_checkpoint = Path(optim_args.denoiser_checkpoint)
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / 'optim_human_human'
    save_dir.mkdir(parents=True, exist_ok=True)
    optim_args.save_dir = save_dir

    diffusion_args = denoiser_args.diffusion_args
    assert 'ddim' in optim_args.respacing
    diffusion_args.respacing = optim_args.respacing
    print('diffusion_args:', asdict(diffusion_args))
    diffusion = create_gaussian_diffusion(diffusion_args)
    sample_fn = diffusion.ddim_sample_loop_full_chain

    # Load dataset
    dataset = SinglePrimitiveDataset(
        cfg_path=vae_args.data_args.cfg_path,
        dataset_path=vae_args.data_args.data_dir,
        sequence_path='./data/stand.pkl',
        batch_size=1,  # Process one person at a time
        device=device,
        enforce_gender='male',
        enforce_zero_beta=1,
    )
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    primitive_utility = dataset.primitive_utility

    # Load joint skin distance
    with open('./data/joint_skin_dist.json', 'r') as f:
        joint_skin_dist = json.load(f)
        joint_skin_dist = torch.tensor(joint_skin_dist, dtype=torch.float32, device=device)
        joint_skin_dist = joint_skin_dist.clamp(min=optim_args.contact_thresh)

    # Load interaction config
    with open(optim_args.interaction_cfg, 'r') as f:
        interaction_cfg = json.load(f)
    
    interaction_name = interaction_cfg['interaction_name'].replace(' ', '_')
    scene_dir = Path(interaction_cfg['scene_dir'])
    
    # Load scene assets
    scene_with_floor_mesh = trimesh.load(scene_dir / 'scene_with_floor.obj', process=False, force='mesh')
    with open(scene_dir / 'scene_sdf.json', 'r') as f:
        scene_sdf_config = json.load(f)
    scene_sdf_grid = np.load(scene_dir / 'scene_sdf.npy')
    scene_sdf_grid = torch.tensor(scene_sdf_grid, dtype=torch.float32, device=device).unsqueeze(0)

    scene_assets = {
        'scene_with_floor_mesh': scene_with_floor_mesh,
        'scene_sdf_grid': scene_sdf_grid,
        'scene_sdf_config': scene_sdf_config,
        'floor_height': interaction_cfg['floor_height'],
    }

    # Setup output path
    out_path = optim_args.save_dir
    filename = f'guidance{optim_args.guidance_param}_seed{optim_args.seed}'
    if optim_args.respacing != '':
        filename = f'{optim_args.respacing}_{filename}'
    if optim_args.zero_noise:
        filename = f'zero_noise_{filename}'
    if optim_args.use_predicted_joints:
        filename = f'use_pred_joints_{filename}'
    filename = f'{interaction_name}_{filename}'
    filename = f'{filename}_interaction{optim_args.weight_interaction}_collision{optim_args.weight_collision}_jerk{optim_args.weight_jerk}'
    out_path = out_path / filename
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize motion for both people
    batch = dataset.get_batch(batch_size=1)
    input_motions, model_kwargs = batch[0]['motion_tensor_normalized'], {'y': batch[0]}
    gender = model_kwargs['y']['gender'][0]
    betas = model_kwargs['y']['betas'][:, :primitive_length, :].to(device)
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': betas[:, 0, :],
        'gender': gender,
    })

    input_motions = input_motions.to(device)
    motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)
    init_history_motion = motion_tensor[:, :history_length, :]

    # Process two-person interaction
    interaction = interaction_cfg['interaction']
    
    # Initialize both people
    history_motion_tensors = []
    transf_rotmats = []
    transf_transls = []
    goal_joints_list = []
    joints_masks = []
    
    for person_idx in range(2):
        history_motion_tensors.append(init_history_motion.clone())
        
        # Set initial position and orientation
        initial_joints = torch.tensor(interaction['people'][person_idx]['init_joints'], 
                                     device=device, dtype=torch.float32)
        transf_rotmat, transf_transl = get_new_coordinate(initial_joints[None])
        transf_rotmats.append(transf_rotmat)
        transf_transls.append(transf_transl)
        
        # Set goal joints (if any)
        if 'goal_joints' in interaction['people'][person_idx]:
            goal_joints = torch.zeros(1, 22, 3, device=device, dtype=torch.float32)
            goal_joints[:, 0] = torch.tensor(interaction['people'][person_idx]['goal_joints'][0], 
                                           device=device, dtype=torch.float32)
            goal_joints_list.append(goal_joints)
            joints_mask = torch.zeros(22, device=device, dtype=torch.bool)
            joints_mask[0] = 1
            joints_masks.append(joints_mask)
        else:
            goal_joints_list.append(None)
            joints_masks.append(torch.zeros(22, device=device, dtype=torch.bool))
    
    # Get text prompts for both people
    text_prompts = [interaction['people'][0]['text_prompt'], 
                   interaction['people'][1]['text_prompt']]
    
    # Optimize motion with interaction constraints
    motion_sequences_list, history_motion_tensors, transf_rotmats, transf_transls = \
        optimize_two_person(history_motion_tensors, transf_rotmats, transf_transls, 
                          text_prompts, goal_joints_list, joints_masks, interaction)
    
    # Save results for both people
    for person_idx in range(2):
        sequence = {
            'texts': motion_sequences_list[person_idx]['texts'],
            'scene_path': scene_dir / 'scene_with_floor.obj',
            'gender': motion_sequences_list[person_idx]['gender'],
            'betas': motion_sequences_list[person_idx]['betas'][0],
            'transl': motion_sequences_list[person_idx]['transl'][0],
            'global_orient': motion_sequences_list[person_idx]['global_orient'][0],
            'body_pose': motion_sequences_list[person_idx]['body_pose'][0],
            'joints': motion_sequences_list[person_idx]['joints'][0],
            'history_length': history_length,
            'future_length': future_length,
            'person_id': person_idx,
            'interaction_type': interaction['interaction_type'],
            'interaction_frame': interaction['interaction_frame'],
        }
        tensor_dict_to_device(sequence, 'cpu')
        with open(out_path / f'person_{person_idx}_sample.pkl', 'wb') as f:
            pickle.dump(sequence, f)

        # Export SMPL sequences
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
            with open(out_path / f'person_{person_idx}_smplx.npz', 'wb') as f:
                np.savez(f, **data_dict)

    print(f'[Done] Results are at [{out_path.absolute()}]')
