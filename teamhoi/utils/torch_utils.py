# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np

from isaacgym.torch_utils import *
import math

@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map

@torch.jit.script
def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map

@torch.jit.script
def quat_to_tan_norm(q):
    # type: (Tensor) -> Tensor
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)
    
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)
    
    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

@torch.jit.script
def euler_xyz_to_exp_map(roll, pitch, yaw):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    q = quat_from_euler_xyz(roll, pitch, yaw)
    exp_map = quat_to_exp_map(q)
    return exp_map

@torch.jit.script
def exp_map_to_angle_axis(exp_map):
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis

@torch.jit.script
def exp_map_to_quat(exp_map):
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q

@torch.jit.script
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta);
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta);

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta;
    ratioB = torch.sin(t * half_theta) / sin_half_theta; 
    
    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q

@torch.jit.script
def calc_heading(q):
    # type: (Tensor) -> Tensor
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def calc_heading_quat(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q

@torch.jit.script
def calc_heading_quat_inv(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q

def sample_polar_no_collision_batch(B, N, R, R_max, d_min, max_trials=100, oversample_factor=2, device="cuda"):
    """
    Sample B batches of N non-colliding 2D points (x, y) in polar space.

    Args:
        B (int): Number of batches to return
        N (int): Points per batch
        R (float): Minimum radius
        R_max (float): Maximum radius
        d_min (float): Minimum distance between any pair in a batch
        max_trials (int): Retry count before failing
        oversample_factor (int): How many more batches to sample at once (B * oversample_factor)

    Returns:
        Tensor of shape (B, N, 2)
    """
    total = B * oversample_factor
    for _ in range(max_trials):
        # Sample candidate batches
        theta = torch.rand(total, N, device=device) * 2 * torch.pi
        r_squared = torch.rand(total, N, device=device) * (R_max**2 - R**2) + R**2
        r = torch.sqrt(r_squared)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        points = torch.stack([x, y], dim=2)  # (total, N, 2)

        # Compute pairwise distances per batch
        diff = points.unsqueeze(2) - points.unsqueeze(1)  # (total, N, N, 2)
        dist_sq = (diff ** 2).sum(dim=-1)  # (total, N, N)
        eye_mask = torch.eye(N, dtype=torch.bool, device=device)
        dist_sq[:, eye_mask] = float('inf')  # ignore self-pairs

        # Check validity
        is_valid = (dist_sq >= d_min ** 2).all(dim=(1, 2))  # (total,)
        valid_points = points[is_valid]  # (valid_count, N, 2)

        if valid_points.shape[0] >= B:
            return valid_points[:B]

    raise RuntimeError(f"Failed to sample {B} valid non-colliding batches after {max_trials} trials.")


def sample_yaw_quaternion_batch(B, N, device="cuda"):
    """
    Sample yaw angles uniformly from [0, 2π] and convert to quaternions (x, y, z, w).

    Args:
        B (int): Number of batches
        N (int): Number of agents per batch
        device (str or torch.device): Torch device

    Returns:
        Tensor of shape (B, N, 4): quaternions representing yaw-only rotation
    """
    yaw = torch.rand(B, N, device=device) * 2 * torch.pi  # (B, N)
    half_yaw = 0.5 * yaw

    quat = torch.zeros(B, N, 4, device=device)
    quat[..., 2] = torch.sin(half_yaw)  # z
    quat[..., 3] = torch.cos(half_yaw)  # w

    return quat

def random_boolean_mask_uniform_random_positions(N_envs, N_max_agents, device='cuda'):
    num_true_per_env = torch.randint(1, N_max_agents + 1, (N_envs,), device=device)
    rand_scores = torch.rand(N_envs, N_max_agents, device=device)
    sorted_scores, _ = rand_scores.sort(dim=1, descending=True)
    thresholds = sorted_scores[torch.arange(N_envs, device=device), num_true_per_env - 1].unsqueeze(1)
    mask = rand_scores >= thresholds  # shape: (N_envs, N_max_agents)
    return mask

def random_boolean_mask_uniform(N_envs, N_max_agents, N_min_agents=1, device='cuda'):
    # Uniformly sample number of agents per environment
    num_true_per_env = torch.randint(N_min_agents, N_max_agents + 1, (N_envs,), device=device)  # (N_envs,)

    # Broadcast and compare against range row
    range_row = torch.arange(N_max_agents, device=device).unsqueeze(0)  # (1, N_max_agents)
    mask = range_row < num_true_per_env.unsqueeze(1)  # (N_envs, N_max_agents)

    return mask

def random_boolean_mask_weighted(N_envs, N_max_agents, weights, device='cuda'):
    """
    Sample a boolean mask for each environment, where the number of active agents
    is chosen according to the given probability weights.

    Args:
        N_envs (int): Number of environments.
        N_max_agents (int): Maximum number of agents per environment.
        weights (list or tensor): Probabilities for choosing k agents (length = N_max_agents).
                                  Example: [0.1, 0.3, 0.4, 0.2] for up to 4 agents.
        device (str): Device to create tensors on.

    Returns:
        mask (torch.BoolTensor): Shape (N_envs, N_max_agents)
    """
    assert len(weights) == N_max_agents, \
        f"Length of weights ({len(weights)}) must match N_max_agents ({N_max_agents})."

    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum()  # normalize

    # sample how many agents to activate in each environment
    choices = torch.arange(1, N_max_agents + 1, device=device)
    num_true_per_env = torch.multinomial(weights, N_envs, replacement=True)
    num_true_per_env = choices[num_true_per_env]  # (N_envs,)

    # broadcast mask
    range_row = torch.arange(N_max_agents, device=device).unsqueeze(0)  # (1, N_max_agents)
    mask = range_row < num_true_per_env.unsqueeze(1)

    return mask

def random_boolean_mask_fixed(N_envs, N_max_agents, N, device='cuda'):

    assert N <= N_max_agents, "N must be <= N_max_agents"
    
    mask = torch.zeros((N_envs, N_max_agents), dtype=torch.bool, device=device)
    mask[:, :N] = True   # always the leftmost N
    
    return mask

def expand_mask_wrt_others_(mask):
    N_envs, N_agents = mask.shape

    # Step 1: Create (N_agents, N_agents - 1) index mask that excludes diagonal
    eye = torch.eye(N_agents, dtype=torch.bool, device=mask.device)  # (N_agents, N_agents)
    other_agent_mask = ~eye  # False on diagonal, True elsewhere
    index_matrix = other_agent_mask.nonzero(as_tuple=False).reshape(N_agents, N_agents - 1, 2)[:, :, 1]  # (N_agents, N_agents - 1)

    # Step 2: Index into the original mask for each environment and agent
    # mask: (N_envs, N_agents), index_matrix: (N_agents, N_agents - 1)
    expanded = mask.unsqueeze(1).expand(-1, N_agents, -1)  # (N_envs, N_agents, N_agents)
    gathered = torch.gather(expanded, 2, index_matrix.unsqueeze(0).expand(N_envs, -1, -1))  # (N_envs, N_agents, N_agents - 1)

    return gathered

@torch.jit.script
def expand_mask_wrt_others(mask: torch.Tensor) -> torch.Tensor:
    N_envs, N_agents = mask.shape

    eye = torch.eye(N_agents, dtype=torch.bool, device=mask.device)
    other_agent_mask = ~eye
    index_flat = torch.nonzero(other_agent_mask)  # TorchScript-compatible
    index_matrix = index_flat.reshape(N_agents, N_agents - 1, 2)[:, :, 1]

    expanded = mask.unsqueeze(1).expand(-1, N_agents, -1)  # (N_envs, N_agents, N_agents)
    gathered = torch.gather(expanded, 2, index_matrix.unsqueeze(0).expand(N_envs, -1, -1))

    return gathered  # (N_envs, N_agents, N_agents - 1)

@torch.jit.script
def quat_rotate_dimflex(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q: (B,4)
    # v: (K,3) or (B,3) or (B,K,3)

    B = q.shape[0]
    q_vec = q[:, :3]           # (B,3)
    q_w  = q[:,  3]            # (B,)

    # Prepare shapes for broadcasting
    squeeze_K = False
    if v.dim() == 2:
        # (N,3): either (B,3) or (K,3)
        if v.shape[0] == B:
            # (B,3) -> treat as (B,1,3), squeeze later
            v = v.unsqueeze(1)         # (B,1,3)
            q_vec = q_vec.unsqueeze(1) # (B,1,3)
            q_w  = q_w.unsqueeze(1)    # (B,1)
            squeeze_K = True
        else:
            # (K,3) -> (1,K,3), broadcast over B
            v = v.unsqueeze(0)         # (1,K,3)
            q_vec = q_vec.unsqueeze(1) # (B,1,3)
            q_w  = q_w.unsqueeze(1)    # (B,1)
    elif v.dim() == 3:
        # (B,K,3)
        q_vec = q_vec.unsqueeze(1)     # (B,1,3)
        q_w  = q_w.unsqueeze(1)        # (B,1)
    else:
        raise RuntimeError("v must be (K,3), (B,3), or (B,K,3)")

    # Now v, q_vec broadcast to (B,*,3); q_w to (B,*,)
    s = (2.0 * q_w * q_w - 1.0).unsqueeze(-1)          # (B,*,1)
    a = v * s                                          # (B,*,3)
    b = 2.0 * q_w.unsqueeze(-1) * torch.cross(q_vec, v, dim=-1)
    c = 2.0 * q_vec * (q_vec * v).sum(-1, keepdim=True)
    out = a + b + c                                    # (B,*,3)

    if squeeze_K:
        out = out.squeeze(1)  # back to (B,3)
    return out

