
import torch
from utils import torch_utils
from utils.torch_utils import quat_rotate_dimflex
import numpy as np

def parse_assets_arg(s: str):
    # Accept "a,b,c", "[a, b, c]" or even with spaces/newlines
    s = s.strip().lstrip("[").rstrip("]")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts

def quat_rotate_broadcast(q, v):
    """
    Rotate vector(s) v by quaternion(s) q.
    q: (..., 4) as (x, y, z, w)
    v: (..., 3)
    returns rotated v of shape (..., 3)
    """
    x, y, z, w = q.unbind(-1)
    # q_vec = (x, y, z)
    # v' = v + 2 * cross(q_vec, cross(q_vec, v) + w * v)
    # Implemented in a batched-friendly way:
    q_vec = torch.stack([x, y, z], dim=-1)
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(q_vec, t, dim=-1)

def rim_cache_single(pts_xyz):
    """
    pts_xyz: (K,3) ordered around the rim (closed loop)
    Returns:
      edge_len: (K,)  length of edge i->i+1 (wrap)
      perimeter: ()   scalar
      s_ccw: (K,)     cumulative arc-length at vertex i (s_ccw[0]=0)
    """
    pts_xy = pts_xyz[..., :2]                          # (K,2)
    # edges i -> i+1 (CCW), with wrap
    nxt = torch.roll(pts_xy, shifts=-1, dims=0)        # (K,2)
    edge_vec = nxt - pts_xy                            # (K,2)
    edge_len = edge_vec.norm(dim=-1)                   # (K,)
    perimeter = edge_len.sum()                         # ()

    return edge_len, perimeter

def divide_envs(num_envs: int, num_assets: int):
    """
    Divide num_envs environments into contiguous ranges for each asset type.

    Args:
        num_envs   : total number of envs (int)
        num_assets : number of different assets (int)

    Returns:
        A list of (start, end) tuples, one per asset. 
        Each covers a contiguous block of env_ids.
    """
    base = num_envs // num_assets
    remainder = num_envs % num_assets

    ranges = []
    start = 0
    for i in range(num_assets):
        # spread the remainder across the first `remainder` chunks
        size = base + (1 if i < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges

def compute_rim_velocity_variance_norm(log_rim, start_t, end_t, env_ids, delta_t=0.0333, eps=1e-12):
    """
    Jerk-focused metric: masked VARIANCE of JERK over time,
    averaged over rim points (K) and axes (x,y,z).

    Args:
        log_rim:  (N_envs, Tmax, K, 3)  positions buffer
        start_t:  (N_envs,)             inclusive start indices (position timeline)
        end_t:    (N_envs,)             exclusive end indices   (position timeline)
        env_ids:  (M,)                  subset of env indices
        delta_t:  float                 dt between frames
        eps:      float

    Returns:
        var_all_env_norm: (M,)  per-env jerkiness metric (jerk variance).
                            (RMS-jerk normalization available but commented)
    """
    # Select env subset
    pos = log_rim[env_ids]                      # (M, Tmax, K, 3)
    start_idx = start_t[env_ids]                # (M,)
    end_idx   = end_t[env_ids]                  # (M,)

    # --- Velocities (T-1) ---
    vel = (pos[:, 1:] - pos[:, :-1]) / float(delta_t)      # (M, T-1, K, 3)
    Tm1 = vel.size(1)

    # Validity on velocity timeline for [start_idx, end_idx)
    t_v = torch.arange(Tm1, device=vel.device)[None, :]    # (1, T-1)
    valid_v = (t_v >= start_idx[:, None]) & (t_v < end_idx[:, None])   # (M, T-1)

    # --- Accelerations (T-2) ---
    acc = (vel[:, 1:] - vel[:, :-1]) / float(delta_t)      # (M, T-2, K, 3)
    valid_a = valid_v[:, 1:] & valid_v[:, :-1]             # (M, T-2)

    # --- Jerks (T-3) ---
    jerk = (acc[:, 1:] - acc[:, :-1]) / float(delta_t)     # (M, T-3, K, 3)

    
    valid_j = valid_a[:, 1:] & valid_a[:, :-1]             # (M, T-3)
    m_j = valid_j[:, :, None, None].to(jerk.dtype).expand_as(jerk)     # (M, T-3, K, 3)

    # --- Masked mean & variance of jerk over time ---
    count = m_j.sum(dim=1)                                  # (M, K, 3)
    sum_jerk = (jerk * m_j).sum(dim=1)                      # (M, K, 3)
    mean_jerk = sum_jerk / count.clamp(min=1.0)             # (M, K, 3)

    jerk_abs = jerk.abs()                               # (M, T-3, K, 3)

    # masked mean of absolute jerk
    mean_abs_jerk = (jerk_abs * m_j).sum(dim=(1, 2, 3)) / m_j.sum(dim=(1, 2, 3)).clamp(min=1.0)  # (M,)

    return mean_abs_jerk

def pairwise_rotation_pos_obs(root_pos, tar_pos, heading_rot, num_humanoids):
    pos_vecs = (root_pos - tar_pos).view(-1, num_humanoids, 3)[:,:,:2]
    normed = pos_vecs / (pos_vecs.norm(2, dim=-1, keepdim=True) + 1e-8)

    # Expand for pairwise comparison
    a = normed.unsqueeze(2)  # (N_envs, N_agents, 1, 2)
    b = normed.unsqueeze(1)  # (N_envs, 1, N_agents, 2)

    dot = (a * b).sum(dim=-1)  # (N_envs, N_agents, N_agents)
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    angles_pairwise = torch.atan2(cross, dot)  # (N_envs, N_agents, N_agents)

    E, A, _ = angles_pairwise.shape
    eye = torch.eye(A, dtype=torch.bool, device=angles_pairwise.device)

    # Boolean-index then reshape: (E, A*A - A) -> (E, A, A-1)
    angles_obs = angles_pairwise[:, eye].view(E, A, 1) # self

    if A > 1:
        angles_others = angles_pairwise[:, ~eye].view(E, A, A - 1)
        angles_obs = torch.cat([angles_obs, angles_others], dim = -1)    
    
    angles_obs = angles_obs.view(-1, num_humanoids).unsqueeze(-1)

    # pos_j - pos_i  -> shape (E, A, A, 3)
    pos = root_pos.view(-1, num_humanoids, 3)
    pos_i = pos[:, :, None, :]   # (E, A, 1, 3)
    pos_j = pos[:, None, :, :]   # (E, 1, A, 3)
    delta_world = pos_j - pos_i  # (E, A, A, 3)

    # Broadcast heading_rot over j dimension: (E, A, 1, 4) -> (E, A, A, 4)
    q = heading_rot.view(-1, num_humanoids, 4)[:, :, None, :].expand(E, A, A, 4)

    # Rotate into each agent i's local heading frame
    delta_local = quat_rotate_broadcast(q, delta_world)  # (E, A, A, 3)
    delta_local_obs = delta_local[:, eye].view(E, A, 1, 3)[:,:,:,:2] # self

    if A > 1:
        delta_local_noself_xy = delta_local[:, ~eye].view(E, A, A - 1, 3)[:,:,:,:2]
        delta_local_obs = torch.cat([delta_local_obs, delta_local_noself_xy], dim=-2)

    delta_local_obs = delta_local_obs.view(-1, num_humanoids, 2)

    other_agents_obs = torch.cat([angles_obs, delta_local_obs], dim = -1)

    return other_agents_obs


def prepare_tensors(
    root_pos,
    root_rot,
    prev_root_pos,
    obj_center,
    hands_pos,
    rim_pts,
    obj_rot,
    normals2d,
    num_agents,
    K_RIM
):
    """
    Prepare flattened/per-agent tensors for reward computation.

    Expected input:
      root_pos:       (N_envs, N_agents, 3)
      root_rot:       (N_envs, N_agents, 4) quaternion (x,y,z,w)
      prev_root_pos:  (N_envs, N_agents, 3)
      obj_center:     (N_envs, 3)
      hands_pos:      (N_envs, N_agents, 2, 3)
      rim_pts:        (N_envs, K_RIM, 3)
      obj_rot:        (N_envs, 4)  quaternion
      normals2d:      (N_envs, K_RIM, 2)
      num_agents      : int. Number of agents per environment.
      K_RIM           : int. Number of rim sample points per object.
                    

    Returns:
      root_pos_all:      (N_envs*N_agents, 3)
      root_rot_all:      (N_envs*N_agents, 4)
      prev_root_pos_all: (N_envs*N_agents, 3)
      obj_center_all:    (N_envs*N_agents, 3)
      hands_pos_all:     (N_envs*N_agents, 2, 3)
      rim_pts_all:       (N_envs*N_agents, K_RIM, 3)
      normals2d_all:     (N_envs*N_agents, M, 2)
      table_normal:      (N_envs, 3)
    """

    # --------------------------------------------------
    # Flatten per-agent tensors
    # --------------------------------------------------
    root_pos_all = root_pos.reshape(-1, root_pos.shape[-1])
    root_rot_all = root_rot.reshape(-1, root_rot.shape[-1])
    prev_root_pos_all = prev_root_pos.reshape(-1, prev_root_pos.shape[-1])
    hands_pos_all = hands_pos.reshape(-1, 2, 3)

    # --------------------------------------------------
    # Repeat object center per agent
    # --------------------------------------------------
    obj_center_all = torch.repeat_interleave(obj_center, num_agents, dim=0)

    # --------------------------------------------------
    # Expand rim points per agent
    # (N_envs, K, 3) -> (N_envs*N_agents, K, 3)
    # --------------------------------------------------
    rim_pts_all = rim_pts[:, None].expand(-1, num_agents, -1, -1).reshape(-1, K_RIM, 3)
    
    # --------------------------------------------------
    # Rotate 2D normals by heading
    # --------------------------------------------------
    heading_rot = torch_utils.calc_heading_quat(obj_rot)
    normals3d = torch.zeros(*normals2d.shape[:-1], 3, device=normals2d.device)
    normals3d[:,:,:2] = normals2d
    normals2d_rotated = quat_rotate_dimflex(heading_rot, normals3d)[:,:,:2]
    normals2d_all = torch.repeat_interleave(normals2d_rotated, num_agents, dim=0)

    # --------------------------------------------------
    return (
        root_pos_all,
        root_rot_all,
        prev_root_pos_all,
        obj_center_all,
        hands_pos_all,
        rim_pts_all,
        normals2d_all,
    )


def draw_circle(gym, viewer, env_ptr, center, radius=0.5, num_segments=32, color=(0,0,0), z_bias=5e-3):
    theta = np.linspace(0, 2*np.pi, num_segments, endpoint=False, dtype=np.float32)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2] + z_bias, dtype=np.float32)  # small lift

    # segments: i -> (i+1)%num_segments
    lines = np.empty((num_segments, 6), dtype=np.float32)
    i2 = (np.arange(num_segments) + 1) % num_segments
    lines[:, 0:3] = np.stack([x, y, z], axis=1)
    lines[:, 3:6] = np.stack([x[i2], y[i2], z[i2]], axis=1)

    col = np.asarray(color, dtype=np.float32)
    if col.max() > 1.0: col = col / 255.0
    cols = np.tile(col[None, :], (num_segments, 1)).astype(np.float32)

    gym.add_lines(viewer, env_ptr, num_segments, lines, cols)
