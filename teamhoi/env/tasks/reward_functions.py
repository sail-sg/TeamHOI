import torch
import math
from utils import torch_utils
from utils.task_util import *
from isaacgym.torch_utils import *
from utils.torch_utils import quat_rotate_dimflex

def compute_walk_reward(root_pos, root_rot, prev_root_pos, obj_center_all, rim_points, normals2d, dt, standing_gap, dist_threshold):
    # encourage the agent to walk towards obj standing points
    near_threshold = 0.04
    pos_err_scale = 2.0
    vel_err_scale = 2.0

    # target speed
    low_speed = 1.5
    high_speed = 2.5

    # compute r_walk_pos
    ## calculate nearest points (root -> rim points)
    P = root_pos[..., :2].unsqueeze(1)                  # (E*A, 1, 2)
    R = rim_points[..., :2]                                # (E*A, K, 2)
    d2 = ((P - R) ** 2).sum(dim=-1)                         # (E*A, K)
    nearest_idx = d2.argmin(dim=1)                          # (E*A,)

    batch_idx = torch.arange(R.size(0), device=R.device)   # (E*A,)
    nearest_points = R[batch_idx, nearest_idx]             # (E*A, 2)


    target_standing_points_pos = nearest_points
    pos_diff = target_standing_points_pos - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1) # n_envs
    pos_err = ((pos_err**0.5) - standing_gap)**2
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    # compute r_walk_vel
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt

    #dir = torch.nn.functional.normalize(pos_diff, dim=-1)
    dir = normals2d[batch_idx, nearest_idx]

    pos_diff2 = obj_center_all[..., 0:2] - root_pos[..., 0:2]
    dir2 = torch.nn.functional.normalize(pos_diff2, dim=-1)

    
    dir_speed = torch.sum(dir * root_vel[..., :2], dim=-1)
    vel_err = torch.relu(low_speed - dir_speed) + torch.relu(dir_speed - high_speed)
    vel_reward = torch.exp(-vel_err_scale * (vel_err * vel_err))
    speed_mask = dir_speed <= 0
    vel_reward[speed_mask] = 0

    # compute r_walk_face
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos[..., 0:3])
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err1 = torch.sum(dir * facing_dir[..., 0:2], dim=-1)
    facing_reward1 = torch.clamp_min(facing_err1, 0.0)
    facing_err2 = torch.sum(dir2 * facing_dir[..., 0:2], dim=-1)
    facing_reward2 = torch.clamp_min(facing_err2, 0.0)


    # compute r_walk
    near_mask = pos_err <= near_threshold
    pos_reward[near_mask] = 1.0
    vel_reward[near_mask] = 1.0

    np_dist = pos_diff.norm(dim=-1)

    k = 10.0  # sharpness of transition
    w = torch.sigmoid(k * (np_dist - dist_threshold))
    facing_reward = (1.0 - w) * facing_reward1 + w * facing_reward2

    return pos_reward, vel_reward, facing_reward, np_dist, root_vel



def compute_angle_reward(root_pos_all, ref_pos_all, mask, n_agents):
    tar_pos = ref_pos_all[..., 0:2]
    root_pos = root_pos_all[..., 0:2]
    pos_vecs = (root_pos - tar_pos).view(-1, n_agents, 2)
    normed = pos_vecs / (pos_vecs.norm(2, dim=-1, keepdim=True) + 1e-8)

    a = normed.unsqueeze(2)  # (N_envs, N_agents, 1, 2)
    b = normed.unsqueeze(1)  # (N_envs, 1, N_agents, 2)

    dot = (a * b).sum(dim=-1)  # (N_envs, N_agents, N_agents)
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    angles_pairwise = torch.atan2(cross, dot)  # (N_envs, N_agents, N_agents)

    E, A, _ = angles_pairwise.shape
    eye = torch.eye(A, dtype=torch.bool, device=angles_pairwise.device)
    two_pi = 2.0 * torch.pi

    D_ccw = (angles_pairwise % two_pi + two_pi) % two_pi
    D_cw  = ((-angles_pairwise) % two_pi + two_pi) % two_pi

    # remove self-pairs
    D_ccw = D_ccw.masked_fill(eye, float('inf'))
    D_cw  = D_cw.masked_fill(eye, float('inf'))

    # --- apply mask ---
    # mask: (E, A) boolean (True if agent is active)
    valid_ij = mask.unsqueeze(1) & mask.unsqueeze(2)  # (E,A,A)
    D_ccw = torch.where(valid_ij, D_ccw, torch.full_like(D_ccw, float('inf')))
    D_cw  = torch.where(valid_ij, D_cw,  torch.full_like(D_cw,  float('inf')))

    # now find nearest neighbors among valid agents only
    ccw_gap, ccw_idx = D_ccw.min(dim=2)   # (E,A)
    cw_gap,  cw_idx  = D_cw.min(dim=2)    # (E,A)

    # handle degenerate rows (no valid neighbors)
    bad = ~torch.isfinite(ccw_gap)
    ccw_gap = torch.where(bad, torch.zeros_like(ccw_gap), ccw_gap)
    cw_gap  = torch.where(bad, torch.zeros_like(cw_gap),  cw_gap)
    ccw_idx = torch.where(bad, torch.full_like(ccw_idx, -1), ccw_idx)
    cw_idx  = torch.where(bad,  torch.full_like(cw_idx, -1), cw_idx)

    angles_err_scale = 3.0

    two_pi = 2.0 * torch.pi

    # number of active agents per env
    m = mask.sum(dim=1) 

    # ideal gap per env
    base = two_pi / m

    # normalized deviations for CCW/CW gaps, zeroed for inactive rows
    diff_ccw = torch.zeros_like(ccw_gap)
    diff_cw  = torch.zeros_like(cw_gap)

    base_expanded = base.unsqueeze(1).expand(-1, n_agents)
    diff_ccw[mask] = (ccw_gap[mask] - (base_expanded[mask])) 
    diff_cw [mask] = (cw_gap [mask] - (base_expanded[mask])) 

    # mean squared error over active agents; average both directions (CCW & CW)
    per_agent_mse = 0.5 * (diff_ccw.pow(2) + diff_cw.pow(2))      # (E,A)

    r_ang = torch.exp(-angles_err_scale * per_agent_mse)            
    r_ang[m==1] = 1
    r_ang = r_ang.view(-1)
    
    return r_ang


def compute_coverage_reward(
    root_pos_all: torch.Tensor,          # (E*A, 3)
    rim_pts_all: torch.Tensor,           # (E*A, K, 3)
    mask: torch.Tensor,                  # (E, A) or (E*A,) -> we will handle both
    obj_held_points_local: torch.Tensor, # (E, K, 2) or (E, K, 3) (we take :2)
    span_xy: torch.Tensor,               # (E, 2)
    n_agents: int,
) -> torch.Tensor:
    """
    Coverage reward r_cov per env.

    Uses per-agent nearest rim vertex -> builds a hull from shifted rim-point pairs
    -> computes nearest hits along 4 directions -> normalizes by span_xy.
    """
    # -----------------------------
    # 1) nearest rim vertex per agent (in XY)
    # -----------------------------
    P = root_pos_all[..., :2].unsqueeze(1)   # (E*A, 1, 2)
    R = rim_pts_all[..., :2]                 # (E*A, K, 2)
    d2 = ((P - R) ** 2).sum(dim=-1)          # (E*A, K)
    nearest_idx = d2.argmin(dim=1)           # (E*A,)

    idx = nearest_idx.view(-1, n_agents)     # (E, A)

    # For invalid agents, replace their idx with the first valid agent's idx in that env
    # (assumes at least one True per env)
    valid_replacement = torch.argmax(mask.int(), dim=1)  # (E,)
    row_ids = torch.arange(idx.shape[0], device=idx.device)
    idx_fallback = idx[row_ids, valid_replacement]       # (E,)
    idx_fallback = idx_fallback.unsqueeze(1).expand(-1, n_agents)  # (E, A)
    idx_valid = torch.where(mask, idx, idx_fallback)     # (E, A)

    # -----------------------------
    # 2) build hull edges from selected rim points
    # -----------------------------
    rim_pts_2d = obj_held_points_local[..., :2]       # (E, K, 2)
    points_b = gather_shifted_pairs(rim_pts_2d, idx_valid, shift=2)  # (E, A*2, 2)
    verts_b, _, _ = hull_vertices_from_directional_extremes_fixed_M(points_b, M=n_agents * 2)  # (E, A*2, 2)
    edges_b = hull_edges_from_vertices(verts_b)          # (E, A*2, 2, 2)

    # -----------------------------
    # 3) nearest hits along 4 dirs from origin, normalize by span_xy
    # -----------------------------
    B = edges_b.shape[0]
    o_b = torch.zeros(B, 2, device=edges_b.device, dtype=torch.float32)

    pt_near, tmin, anyhit = nearest_hits_4dirs(o_b, edges_b, miss_value=0)  # (E, 4, 2) typically
    d_pt_near = pt_near.norm(dim=-1)                                        # (E, 4)
    d_pt_near = d_pt_near.view(-1, 2, 2)                                    # (E, 2, 2) -> [x,-x],[y,-y]

    r_cov = d_pt_near / span_xy.unsqueeze(-1)                               # (E, 2, 2)
    r_cov = torch.min(r_cov, dim=-1)[0]                                     # (E, 2)
    r_cov = r_cov.mean(dim=-1)                                              # (E,)

    # If only one valid agent, define full coverage
    n_agents_per_env = mask.sum(dim=-1)                                     # (E,)
    r_cov[n_agents_per_env == 1] = 1.0

    # all agents share the same reward
    r_cov = torch.repeat_interleave(r_cov, repeats=n_agents, dim=0)

    return r_cov


def compute_hands_and_lifts_reward(hands_pos, target_pts, np_dist, dist_threshold = 3.0, proxy_scale = 5.0, held_target_z = 0.94):
    """
    hands_pos: (B, 2, 3)   -> positions of left/right hands
    target_pts: (B, K, 3)  -> rim points for each env
    """
    B = hands_pos.shape[0]
    K = target_pts.shape[1]

    target_sep = 0.4
    sep_scale = 5.0
    prox_scale = proxy_scale
    height_match_scale = 20.0
    z_scale = 5.0

    # --- Nearest rim point per hand ---
    # Distances from each hand to all rim points: (B, 2, K)
    dists = torch.norm(hands_pos.unsqueeze(2) - target_pts.unsqueeze(1), dim=-1)
    nearest_dist, nn_idx = dists.min(dim=-1)                             # (B,2)

    # Gather nearest rim points per hand: (B,2,3)
    tp_exp = target_pts.unsqueeze(1).expand(B, 2, K, 3)                  # (B,2,K,3)
    nn_idx_exp = nn_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 2, 1, 3)
    nn_pts = torch.gather(tp_exp, 2, nn_idx_exp).squeeze(2)              # (B,2,3)

    # --- Proximity reward (per hand), then average ---
    hand_prox_reward1 = torch.exp(-prox_scale * nearest_dist)             # (B,2)
    prox_reward = hand_prox_reward1.mean(dim=-1)                          # (B,)

    # --- XY separation reward for hand pair ---
    hand_sep = (hands_pos[:, 0, :2] - hands_pos[:, 1, :2]).norm(dim=-1)  # (B,)
    base = target_sep
    low = 1.0 * base
    high = 1.5 * base
    sep_err = torch.relu(low - hand_sep) + torch.relu(hand_sep - high)   # (B,)
    sep_reward = torch.exp(-sep_scale * sep_err * sep_err)               # (B,)

    # --- Same-height reward across hands ---
    z_diff = hands_pos[:, 0, 2] - hands_pos[:, 1, 2]                     # (B,)
    same_height_reward = torch.exp(-height_match_scale * (z_diff * z_diff))

    # --- z_above (1 when angle >= 90°, decay to 0 as it approaches 0°) ---
    eps=1e-8
    z_softness = 3.0

    # Upward world vector
    up = torch.tensor([0.0, 0.0, 1.0], device=hands_pos.device, dtype=hands_pos.dtype)

    v = hands_pos - nn_pts                                 # (B,2,3)
    v_norm = v.norm(dim=-1).clamp_min(eps)                 # (B,2)

    # Dot with world-up
    cos_theta = (v * up).sum(dim=-1) / v_norm              # (B,2)
    # keep 1 for cos<=0; for cos>0 decay smoothly via exp(-k * cos)
    z_above_per_hand = torch.where(cos_theta <= 0.0,
                                   torch.ones_like(cos_theta),
                                   torch.exp(-z_softness * cos_theta))
    z_above = z_above_per_hand.mean(dim=-1)    

    hand_reward = prox_reward * z_above * sep_reward * same_height_reward

    mask_outside = np_dist > dist_threshold
    hand_reward[mask_outside] = 0.0

    # contact
    contact_threshold = 0.04
    activate_threshold = 0.06

    contact_score = (1.0 - nearest_dist / activate_threshold).clamp(min=0.0)  # (B,2)
    #contact_reward = contact_score.mean(dim=-1)   # (B,)
    contact_reward = contact_score.min(dim=-1)[0]

    held_z_err = (nn_pts[..., 2] - held_target_z).abs()
    held_per_hand = torch.exp(-z_scale * held_z_err)
    contact_mask = (nearest_dist < contact_threshold).to(hands_pos.dtype)  

    #denom = contact_mask.sum(dim=-1).clamp_min(1.0)          # (B,)
    held_point_height_reward = (held_per_hand * contact_mask).sum(dim=-1) / 2
    held_point_height_reward = torch.where(contact_mask.sum(dim=-1) > 0, held_point_height_reward, torch.zeros_like(held_point_height_reward))

    return hand_reward, contact_reward, held_point_height_reward, contact_mask, nearest_dist

def compute_transport_reward(obj_pos, target_pos, two_hands_touch):
    d = target_pos[..., :2] - obj_pos[..., :2]
    e = (d ** 2).sum(dim=-1)
    r = torch.exp(-0.15 * e)
    r[~two_hands_touch] = 0.0
    return r

def compute_align_reward(root_pos_all, root_rot_all, valid_humans, obj_pos, target_pos, two_hands_touch, n_agents):
    root_xyz = root_pos_all.view(-1, n_agents, 3)
    root_rot = root_rot_all.view(-1, n_agents, 4)
    target_xyz = target_pos

    dists = (root_xyz - target_xyz.unsqueeze(1)).norm(dim=-1)         # (B, A)
    dists[~valid_humans] = -99

    idx = dists.argmax(dim=-1)                                    # (B,)
    B, A = root_rot.shape[:2]
    root_rot_nearest = root_rot[torch.arange(B, device=idx.device), idx]

    heading_rot = torch_utils.calc_heading_quat(root_rot_nearest)

    facing_dir = torch.zeros_like(heading_rot[..., 0:3])
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)

    target_diff = target_xyz[..., :2] - obj_pos[..., :2]
    target_dir = torch.nn.functional.normalize(target_diff, dim=-1)
    facing_err = torch.sum(target_dir * facing_dir[..., 0:2], dim=-1)
    align_reward = torch.clamp_min(facing_err, 0.0)
    
    close = (target_diff).norm(dim=-1) < 0.5
    align_reward[close] = 1.0
    align_reward[~two_hands_touch] = 0.0

    return align_reward

def compute_putdown_reward(
    putdown_mask,
    num_agents,
    hands_pos_all,
    hands_dist,
    root_vel,
    hand_target_z=0.65,
    k=5.0
):
    # --- masks ---
    reached = torch.repeat_interleave(putdown_mask, repeats=num_agents, dim=0)  # (B,)
    no_touch = hands_dist.amin(dim=-1) > 0.07
    end = reached & no_touch

    # --- hand height reward ---
    dz = (hands_pos_all[..., 2] - hand_target_z).abs()   # (B, 2)
    r_putdown = torch.exp(-k * dz).amin(dim=1)           # (B,)

    # --- zero velocity reward ---
    vel_xy = root_vel[:, :2].norm(dim=-1)
    zero_vel_reward = torch.exp(-2.0 * vel_xy) * reached

    # --- finalize ---
    r_putdown = r_putdown * reached
    r_putdown = r_putdown.masked_fill(end, 1.0)
    r_putdown = 0.8 * r_putdown + 0.2 * zero_vel_reward

    return r_putdown, reached

def hull_vertices_from_directional_extremes_fixed_M(points_b: torch.Tensor, M: int = 16, n_probe: int = 64):
    """
    Fully vectorized. Returns exactly M vertices per batch, deduped then padded.
    points_b: (B, N, 2)
    M:        desired number of vertices/edges
    n_probe:  number of sweep directions (>= M recommended)
    Returns:
      verts_b: (B, M, 2)  CCW sweep order; padded by repeats if needed
      idx_b:   (B, M)     indices into original points
      valid_b: (B, M)     True where a slot is a unique extreme; False = padded/duplicate
    """
    B, N, _ = points_b.shape
    device, dtype = points_b.device, points_b.dtype

    thetas = torch.linspace(0.0, 2*math.pi, steps=n_probe, device=device, dtype=dtype)
    U = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=0)               # (2, T)
    proj = torch.einsum('bnc,cm->bnm', points_b, U)                               # (B,N,T)
    idx_dir = proj.argmax(dim=1)                                                  # (B,T) -> point index per direction

    T = n_probe
    dir_range = torch.arange(T, device=device)                                    # (T,)

    # matches[b,n,t] True if point n is the extreme at direction t
    matches = (idx_dir.unsqueeze(1) == torch.arange(N, device=device).view(1, -1, 1))  # (B,N,T)
    # first occurrence position per point (T means "never occurs")
    pos_candidates = torch.where(matches, dir_range.view(1,1,-1), torch.full((1,1,T), T, device=device))
    first_pos, _ = pos_candidates.min(dim=2)                                      # (B,N)

    # sort points by first occurrence, then take first M
    sort_pos, sort_idx = first_pos.sort(dim=1)                                    # (B,N)
    sel_pos  = sort_pos[:, :M]                                                    # (B,M)
    sel_idx  = sort_idx[:, :M]                                                    # (B,M)
    valid    = sel_pos < T                                                         # (B,M)

    # pad by repeating the last valid index in each batch
    num_valid       = valid.sum(dim=1, keepdim=True).clamp_min(1)                 # (B,1)
    last_valid_slot = (num_valid - 1)                                             # (B,1)
    last_valid_idx  = torch.gather(sel_idx, 1, last_valid_slot)                   # (B,1)
    sel_idx_padded  = torch.where(valid, sel_idx, last_valid_idx.expand(-1, M))   # (B,M)

    b_ix   = torch.arange(B, device=device).unsqueeze(1).expand(B, M)             # (B,M)
    verts  = points_b[b_ix, sel_idx_padded, :]                                    # (B,M,2)

    return verts, sel_idx_padded, valid

def hull_edges_from_vertices(verts_b: torch.Tensor):
    verts_next = torch.roll(verts_b, -1, dims=1)
    return torch.stack([verts_b, verts_next], dim=2)                              # (B,M,2,2)

def cross2(a, b):
    # 2D scalar cross product: a_x*b_y - a_y*b_x
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]

def ray_segment_intersections_4dirs(
    o_b: torch.Tensor,        # (B,2) ray origins
    edges_b: torch.Tensor,    # (B,M,2,2) segments [a,b]
    miss_value: float = float('nan'),
    eps: float = 1e-12,
):
    """
    Intersect, per batch, 4 rays with M segments:
      Rays are fixed: [[1,0],[0,1],[-1,0],[0,-1]] (x+, y+, x-, y-)
    Returns:
      pts_all: (B,4,M,2) intersection points (miss_value on miss)
      hit:     (B,4,M)   hit mask
      t:       (B,4,M)   ray parameter (inf on miss)
      s:       (B,4,M)   segment parameter (nan on miss)
    """
    B, M = edges_b.shape[:2]
    device = edges_b.device
    dtype  = edges_b.dtype

    # 4 directions, shared across batch
    U4 = torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]], device=device, dtype=dtype)  # (4,2)

    a = edges_b[..., 0, :]                 # (B,M,2)
    b = edges_b[..., 1, :]                 # (B,M,2)
    v = b - a                              # (B,M,2)

    # Broadcast: rays (B,4,1,2), edges (B,1,M,2)
    o = o_b[:, None, None, :].expand(B, 4, 1, 2)   # (B,4,1,2)
    u = U4[None, :, None, :].expand(B, 4, 1, 2)    # (B,4,1,2)
    A = a[:, None, :, :].expand(B, 4, M, 2)        # (B,4,M,2)
    V = v[:, None, :, :].expand(B, 4, M, 2)        # (B,4,M,2)

    det = cross2(u, V)                              # (B,4,M)
    parallel = det.abs() < eps
    denom = torch.where(parallel, torch.ones_like(det), det)

    AO = A - o                                      # (B,4,M,2)
    t = cross2(AO, V) / denom                       # (B,4,M)
    s = cross2(AO, u) / denom                       # (B,4,M)

    hit = (~parallel) & (t >= 0) & (s >= 0) & (s <= 1)

    pts_hit = o + t[..., None] * u                  # (B,4,M,2)
    pts = torch.full_like(pts_hit, miss_value)
    pts = torch.where(hit[..., None], pts_hit, pts)

    # Clean t/s where miss
    t = torch.where(hit, t, torch.full_like(t, float('inf')))
    s = torch.where(hit, s, torch.full_like(s, float('nan')))

    return pts, hit, t, s

def nearest_hits_4dirs(
    o_b: torch.Tensor,        # (B,2)
    edges_b: torch.Tensor,    # (B,M,2,2)
    miss_value: float = float('nan'),
):
    """
    Nearest intersection along each of the 4 fixed directions per batch.
    Returns:
      pt:     (B,4,2) nearest hit per direction (miss_value if none)
      tmin:   (B,4)   nearest t (inf if none)
      anyhit: (B,4)   boolean per direction
    """
    pts_all, hit_all, t_all, _ = ray_segment_intersections_4dirs(o_b, edges_b, miss_value=miss_value)
    # min over edges (dim=2)
    tmin, idx = t_all.min(dim=2)                     # (B,4), (B,4)
    anyhit = torch.isfinite(tmin)

    # gather nearest points along edge dim
    B = edges_b.shape[0]
    gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(B, 4, 1, 2)  # (B,4,1,2)
    pt = torch.gather(pts_all, dim=2, index=gather_idx).squeeze(2)   # (B,4,2)

    # fill miss_value where no hit
    pt = torch.where(anyhit.unsqueeze(-1), pt, torch.full_like(pt, miss_value))
    return pt, tmin, anyhit


def gather_shifted_pairs(points_b: torch.Tensor, idx: torch.Tensor, shift: int = 2):
    """
    Gather neighbors for given indices with wrap-around shift.
    
    Args:
      points_b: (B,K,2)
      idx:      (B,M) indices (already sampled)
      shift:    neighbor offset
      
    Returns:
      out: (B,2M,2) points (shifted -shift and +shift for each idx)
    """
    B, K, _ = points_b.shape
    M = idx.shape[1]

    idx_minus = (idx - shift) % K
    idx_plus  = (idx + shift) % K
    idx2 = torch.cat([idx_minus, idx_plus], dim=1)  # (B,2M)

    b_ix = torch.arange(B, device=points_b.device).unsqueeze(1).expand(B, 2*M)
    out = points_b[b_ix, idx2]  # (B,2M,2)
    return out