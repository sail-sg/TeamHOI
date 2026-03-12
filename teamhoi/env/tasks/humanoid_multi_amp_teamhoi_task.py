import os
import json
import torch
import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *

import env.tasks.humanoid_multi_amp_task as humanoid_amp_task_multi
from utils import torch_utils
import yaml
from utils.torch_utils import quat_rotate_dimflex
import torch.nn.functional as F
import pandas as pd
from env.tasks.reward_functions import *
from utils.task_util import *


class HumanoidAMPTableLift(humanoid_amp_task_multi.HumanoidAMPTaskMulti):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.num_envs = cfg["env"]["numEnvs"]
        
        train_cfg_path = cfg['args'].cfg_train
        self.is_train = not cfg['args'].test

        assets_list = parse_assets_arg(cfg['args'].assets)    
        
        # Open and load
        with open(train_cfg_path, "r") as f:
            train_cfg = yaml.safe_load(f)
        
        self.dist_threshold = float(train_cfg['params']['config'].get("dist_threshold", 10000.0))
        self._target_dist_min = 3 
        self._target_dist_max = 10

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.K_RIM = 64
        asset_cfg_paths = [os.path.join("assets", name + ".json") for name in assets_list]

        self.assets = []
        self.asset_cfgs = []
        for cfg_path in asset_cfg_paths:
            with open(cfg_path, "r") as f:
                asset_cfg = json.load(f)
            self.asset_cfgs.append(asset_cfg)
            self.assets.append(asset_cfg["asset"])

        self.ranges = divide_envs(self.num_envs, len(self.assets))
        self.rim_points_list = []
        self.obj_spans = []

        self._build_rim_points(device)
        self.build_rim_cache() # create self.perimeter and self.s_ccw
        self.build_obj_points_and_normals() # create self.obj_held_points_local and self.normals2d
        self.num_held_points = self.K_RIM
        self.obj_height = 0.82 # asume all same

        self.self_obs_dim = [
            223, # proprio
            3, # obj_root
            self.num_held_points * 3,
            6, # nn_pts
            3 # goal
        ]
        self.task_obs_dim = [3,
                            self.num_held_points * 3,
                            6,
                            3, # goal
                            cfg['env_max_humanoids'] * (3+6) # relative angles and pos
                             ]
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._prev_root_pos_tensor = torch.zeros([self.num_envs, self.num_agents, 3], device=self.device, dtype=torch.float)

        lift_body_names = cfg["env"]["liftBodyNames"]
        self._lift_body_ids = self._build_lift_body_ids_tensor(lift_body_names)

        # Toggle for viz and eval ###############
        self.viz = False
        self.log_metrics = False
        ##################################
        self.first_call_done = False # used in log metrics

        self._build_obj_tensors()
        self._build_target_state_tensors()
        self.dist_to_nearest_point = torch.ones(self.num_envs * self.num_agents).to(self.device) * 10
        self.w = train_cfg['params']['config']['w']
        self.target_height = 0.94
        
        if self.viz == True:
            self.t = 0
            self.T = 550
            self.motion_agents = torch.zeros(self.T, self.num_envs, self.num_agents, 15, 7, device = device,dtype=torch.float32)
            self.motion_obj = torch.zeros(self.T, self.num_envs, 7, device = device,dtype=torch.float32)

    def _build_lift_body_ids_tensor(self, lift_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles_list[0][0]
        body_ids = []

        for body_name in lift_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_rim_points(self, device="cpu"):
        
        for asset_cfg in self.asset_cfgs:
            typ = asset_cfg["type"]

            if typ == "rect":
                full_x = asset_cfg["hx"]
                full_y = asset_cfg["hy"]
                offset = asset_cfg["offset"]
                K_RIM = self.K_RIM
                z_local = asset_cfg["z_local"]

                hx = max(0.5 * full_x - offset, 0.0)
                hy = max(0.5 * full_y - offset, 0.0)

                perim = 4 * (hx + hy)
                s_vals = torch.arange(0, K_RIM, device=device, dtype=torch.float32) * (perim / K_RIM)

                pts_local = torch.zeros((K_RIM, 3), dtype=torch.float32, device=device)

                # bottom edge
                mask = s_vals < 2 * hx
                s = s_vals[mask]
                pts_local[mask, 0] = -hx + s
                pts_local[mask, 1] = -hy

                # right edge
                mask = (s_vals >= 2 * hx) & (s_vals < 2 * hx + 2 * hy)
                s = s_vals[mask] - 2 * hx
                pts_local[mask, 0] = +hx
                pts_local[mask, 1] = -hy + s

                # top edge
                mask = (s_vals >= 2 * hx + 2 * hy) & (s_vals < 4 * hx + 2 * hy)
                s = s_vals[mask] - (2 * hx + 2 * hy)
                pts_local[mask, 0] = +hx - s
                pts_local[mask, 1] = +hy

                # left edge
                mask = s_vals >= (4 * hx + 2 * hy)
                s = s_vals[mask] - (4 * hx + 2 * hy)
                pts_local[mask, 0] = -hx
                pts_local[mask, 1] = +hy - s

                pts_local[:, 2] = z_local

                span_xy = [hx, hy]

            elif typ == "round":
                offset = asset_cfg["offset"]
                TABLE_RADIUS = asset_cfg["table_radius"]
                K_RIM = self.K_RIM
                z_local = asset_cfg["z_local"]

                r = TABLE_RADIUS - offset
                angles = (2.0 * torch.pi) * torch.arange(K_RIM, device=device, dtype=torch.float32) / K_RIM
                pts_local = torch.stack([
                    r * torch.cos(angles),
                    r * torch.sin(angles),
                    torch.full((K_RIM,), z_local, device=device, dtype=torch.float32)
                ], dim=1)

                span_xy = [r, r]

            else:
                raise ValueError(f"Unknown asset type: {typ}")

            self.rim_points_list.append(pts_local)
            self.obj_spans.append(span_xy)
    
    def build_rim_cache(self):
        """
        Build expanded rim caches for all envs.
        """
        spans = []

        for asset_idx, pts_local in enumerate(self.rim_points_list):
            start, end = self.ranges[asset_idx]
            num = end - start

            span = torch.as_tensor(self.obj_spans[asset_idx], device=pts_local.device, dtype=torch.float32)  # (2,)
            spans.append(span.unsqueeze(0).repeat(num, 1))              # (num, 2)

        # concat into env-major arrays
        self.span_xy   = torch.cat(spans, dim=0)           # (num_envs, 2)
    
    def build_obj_points_and_normals(self):
        """
        Build expanded obj_held_points_local and normals2d for all envs.
        - self.obj_held_points_local : (num_envs, K, 3)
        - self.normals2d             : (num_envs, K, 2)
        """
        obj_pts_all = []
        normals_all = []

        for asset_idx, pts_local in enumerate(self.rim_points_list):

            # --- obj_held_points_local
            obj_pts = pts_local.unsqueeze(0)  # (1, K, 3)

            # --- normals2d
            xy = pts_local[:, :2]                       # (K, 2)
            prev_xy = torch.roll(xy, 1, dims=0)
            next_xy = torch.roll(xy, -1, dims=0)
            tangent = next_xy - prev_xy                 # (K, 2)
            tangent = F.normalize(tangent, dim=-1)      # unit tangent
            normals_2d = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
            normals_2d = F.normalize(normals_2d, dim=-1)  # (K, 2)
            normals = normals_2d.unsqueeze(0)             # (1, K, 2)

            # --- expand for each env in range
            start, end = self.ranges[asset_idx]
            num = end - start

            obj_pts_all.append(obj_pts.repeat(num, 1, 1))     # (num, K, 3)
            normals_all.append(normals.repeat(num, 1, 1))     # (num, K, 2)

        # concat envs
        self.obj_held_points_local = torch.cat(obj_pts_all, dim=0)  # (num_envs, K, 3)
        self.normals2d             = torch.cat(normals_all, dim=0) # (num_envs, K, 2)

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._obj_handles = []
        self._load_obj_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return    

    def _load_obj_asset(self):
        asset_root = "assets"
        self._obj_asset = []

        loaded_assets = []
        for asset_file in self.assets:
            full_path = os.path.join(asset_root, asset_file)
            assert os.path.exists(full_path), f"URDF not found at {full_path}"

            opts = gymapi.AssetOptions()
            opts.fix_base_link = False
            opts.collapse_fixed_joints = True

            asset = self.gym.load_asset(self.sim, asset_root, asset_file, opts)

            # simple shape props
            shape_props = self.gym.get_asset_rigid_shape_properties(asset)
            for i, p in enumerate(shape_props):
                p.friction = 1.0
                if i == 0:  # first shape (e.g. tabletop)
                    p.friction = 1.0
                    p.restitution = 0.0
                    p.rolling_friction = 0.0
                    p.torsion_friction = 0.0
            self.gym.set_asset_rigid_shape_properties(asset, shape_props)

            loaded_assets.append(asset)

        # assign assets to envs per ranges
        for asset_idx, (start, end) in enumerate(self.ranges):
            asset = loaded_assets[asset_idx]
            for _ in range(start, end):
                self._obj_asset.append(asset)

        assert len(self._obj_asset) == self.num_envs

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        if self.is_train:
            self._build_fake_ground(env_id, env_ptr)
        self._build_obj(env_id, env_ptr)

        return
    
    def _build_obj(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.z = 10

        obj_handle = self.gym.create_actor(env_ptr, self._obj_asset[env_id], default_pose, "obj", col_group, col_filter, segmentation_id)
        props = self.gym.get_actor_dof_properties(env_ptr, obj_handle)
        props['friction'].fill(1.0)
        self.gym.set_actor_dof_properties(env_ptr, obj_handle, props)
        self._obj_handles.append(obj_handle)
        return

    def _build_fake_ground(self, env_id, env_ptr):
        # Ground parameters
        ground_size_x = 24.0  # width
        ground_size_y = 24.0  # length
        ground_thickness = 0.1  # height of the obj
        ground_height = 5    # where you want the top surface of the ground to be

        # Create the ground asset only once
        if not hasattr(self, 'ground_asset'):
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True  # make it static
            self.ground_asset = self.gym.create_box(
                self.sim, ground_size_x, ground_size_y, ground_thickness, asset_options)

        # Set the pose so the top of the box aligns with `ground_height`
        pose = gymapi.Transform()
        pose.p.z = ground_height - ground_thickness / 2.0

        # Add the box as an actor
        self.gym.create_actor(env_ptr, self.ground_asset, pose, f"fake_ground_{env_id}", env_id, 0, 0)

    def _build_target_state_tensors(self):
        self._target_pos = torch.zeros(self.num_envs, 3).to(self.device)
        self._target_rot = torch.zeros(self.num_envs, 4).to(self.device)
        self._target_end_pos = torch.zeros(self.num_envs, 3).to(self.device)

        return
    
    def _build_obj_tensors(self):
        num_actors = self.get_num_actors_per_env()

        self._obj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., -1, :]

        if self.is_train:
            self._obj_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + self.num_agents + 1
        else:
            self._obj_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + self.num_agents


        self.obj_held_points = torch.zeros(self.num_envs, self.K_RIM, 3).to(self.device)
        self.putdown_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self.log_metrics == True:
            self.steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.start_t = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.end_t =  torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.start_lift = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            self.log_rim = torch.zeros(self.num_envs, 1200, self.K_RIM, 3, device=self.device)

            self.t_coop = torch.zeros(self.num_envs, device=self.device)
            self.d_last = torch.zeros(self.num_envs, device=self.device)

    def _reset_actors(self, env_ids):

        if self.log_metrics == True:
            if env_ids is not None and self.first_call_done == True:
                var_all_env_norm = compute_rim_velocity_variance_norm(
                    self.log_rim, self.start_t, self.end_t, env_ids, delta_t=0.0333
                )

                t_coop = self.t_coop[env_ids] / (self.end_t[env_ids] - self.start_t[env_ids])
                t_coop = t_coop.clamp(max=1.0, min=0.0)

                df = pd.DataFrame({
                    "env_id": env_ids.cpu().numpy(),
                    "steps": self.steps[env_ids].detach().cpu().numpy(),
                    "start_t": self.start_t[env_ids].int().detach().cpu().numpy(),
                    "end_t": self.end_t[env_ids].int().detach().cpu().numpy(),
                    "target_reached": self.putdown_mask[env_ids].int().detach().cpu().numpy(),
                    "d_end": self.d_last[env_ids].detach().cpu().numpy(),
                    "jerk": var_all_env_norm.detach().cpu().numpy(),
                    "t_coop": t_coop.detach().cpu().numpy()
                })
                df.to_csv(
                    "log_out.csv",
                    mode="a",
                    index=False,
                    header=False
                )
            
            if env_ids is not None:
                self.steps[env_ids] = 0
            
            self.first_call_done = True

            # reset self.start_t and self.start_lift
            self.start_t[env_ids] = 0
            self.start_lift[env_ids] = False
            self.end_t[env_ids] = 0
            self.log_rim[env_ids] = 0.0
            self.t_coop[env_ids] = 0.0

        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        self._reset_obj(env_ids)

        if self.viz == True:
            self.t = 0

        return
    
    def _reset_obj(self, env_ids):

        n = len(env_ids)

        self._obj_states[env_ids,0] = 0.0 
        self._obj_states[env_ids,1] =  0.0
        self._obj_states[env_ids, 2] = self.obj_height + 0.03
        self._obj_states[env_ids, 3:7] = torch.tensor([.0, 0.0, 0.0, 1.0], dtype=self._obj_states.dtype, device=self._obj_states.device)
        self._obj_states[env_ids, 7:] = 0.0

        return

    def _reset_target(self, env_ids):

        n = len(env_ids)

        self.target_height = 0.94

        random_numbers_rot = torch.rand([n], dtype=self._target_end_pos.dtype, device=self._target_end_pos.device)
        rand_rot_theta = 2 * np.pi * random_numbers_rot
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_pos.dtype, device=self._target_pos.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)
        self._target_rot[env_ids] = rand_rot

        rand_dist = (self._target_dist_max - self._target_dist_min) * torch.rand(
            [n], dtype=self._target_end_pos.dtype, device=self._target_end_pos.device) + self._target_dist_min
        random_numbers = torch.rand(
            [n], dtype=self._target_end_pos.dtype, device=self._target_end_pos.device)

        rand_theta = 2 * np.pi * random_numbers
        self._target_end_pos[env_ids, 0] = rand_dist * torch.cos(rand_theta) 
        self._target_end_pos[env_ids, 1] = rand_dist * torch.sin(rand_theta)
        self._target_end_pos[env_ids, 2] = self.target_height

        self.putdown_mask[env_ids] = False
        
        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        ids = [self._obj_actor_ids[env_ids]]
        reset_ids = torch.cat(ids, dim=0).to(torch.int32).contiguous()

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(reset_ids),
            reset_ids.numel()
        )

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        self._prev_root_pos_tensor = self._humanoid_root_states_tensor[:,:,0:3].clone()
        
        return
    
    def update_held_points(self, obj_states, env_ids=None):
        pt = self.obj_held_points_local  # (K,3)

        if env_ids is None:
            obj_pos = obj_states[..., 0:3]   # (B,3)
            obj_rot = obj_states[..., 3:7]   # (B,4)

            # rotate K points for each env -> (B,K,3), then translate by obj_pos
            rotated = quat_rotate_dimflex(obj_rot, pt)                 # (B,K,3)
            self.obj_held_points[:] = obj_pos.unsqueeze(1) + rotated  # (B,K,3)
        else:
            obj_pos = obj_states[env_ids, 0:3]  # (b,3)
            obj_rot = obj_states[env_ids, 3:7]  # (b,4)

            rotated = quat_rotate_dimflex(obj_rot, pt[env_ids])                      # (b,K,3)
            self.obj_held_points[env_ids] = obj_pos.unsqueeze(1) + rotated

        return

    def _compute_task_obs_tensor(self, env_ids=None):

        if env_ids is None or len(env_ids) == self.num_envs:
            root_states = self._humanoid_root_states_tensor
        else:
            root_states = self._humanoid_root_states_tensor[env_ids]

        num_envs, num_agents, _ = root_states.shape
        root_states = root_states.reshape(-1, root_states.shape[2])

        self.update_held_points(self._obj_states, env_ids)

        obj_pos = self._obj_states[env_ids][..., :3]
        obj_pos = obj_pos.unsqueeze(1).expand(-1, num_agents, -1)
        obj_pos = obj_pos.reshape(-1, obj_pos.shape[2])

        obj_held_pts = self.obj_held_points[env_ids] # B, K, 3
        obj_held_pts = obj_held_pts.unsqueeze(1).expand(-1, num_agents, -1, -1)
        obj_held_pts = obj_held_pts.reshape(-1, obj_held_pts.size(2), obj_held_pts.size(3))

        hand_states = self._rigid_body_pos_tensor[env_ids][:, :, self._lift_body_ids, :].clone() # (n_envs, n_agents, 2, 3)
        hand_states = hand_states.reshape(-1, hand_states.shape[2], hand_states.shape[3])
        
        # 1) Pairwise diffs and distances: (B,2,K,3) -> (B,2,K)
        diff  = hand_states.unsqueeze(2) - obj_held_pts.unsqueeze(1)   # (B,2,K,3)
        dists = diff.norm(dim=-1)                                      # (B,2,K)

        # 2) Nearest rim index per hand: (B,2)
        nn_idx = dists.argmin(dim=-1)

        # 3) Gather nearest rim points per hand: (B,2,3)
        pts_exp = obj_held_pts.unsqueeze(1).expand(-1, 2, -1, 3)       # (B,2,K,3)
        idx_exp = nn_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 2, 1, 3)
        nn_pts  = torch.gather(pts_exp, 2, idx_exp).squeeze(2)         # (B,2,3)

        # 4) Vector from rim -> hand and flatten to 6 dims
        vec_rim_to_hand = hand_states - nn_pts                         # (B,2,3)
        obs_vec6 = vec_rim_to_hand.reshape(vec_rim_to_hand.size(0), -1)  # (B,6)

        # Add lift target
        tar_pos = self._target_end_pos[env_ids]

        nn_pts_z = nn_pts[:,:,2].mean(dim=1)

        # Target-object distance
        delta_xy = self._target_end_pos[env_ids][...,:2] - self._obj_states[env_ids][..., :2]
        d = delta_xy.norm(dim=-1)

        near = d < 0.03
        self.putdown_mask[env_ids] |= near     # in-place OR

        idx = env_ids[self.putdown_mask[env_ids]]
        tar_pos[idx, 2] = 0.78

        tar_pos = tar_pos.unsqueeze(1).expand(-1, num_agents, -1)
        tar_pos = tar_pos.reshape(-1, tar_pos.shape[2])

        hand_z = hand_states[:,:,2].max(dim=-1)[0]

        obs = compute_obs(root_states, obj_pos, obj_held_pts, obs_vec6, tar_pos, hand_z, self.putdown_mask[env_ids], num_humanoids = self.num_agents)

        if self.viz == True:
            if self.t < self.T:
                motion = torch.cat([self._rigid_body_pos_tensor, self._rigid_body_rot_tensor], dim=-1)
                self.motion_agents[self.t] = motion
                self.motion_obj[self.t] = self._obj_states[:,:7]
            elif self.t == self.T:
                output_dict = {}
                output_dict['motion_agents'] = self.motion_agents[:self.T].detach().cpu().numpy()
                output_dict['motion_object'] = self.motion_obj[:self.T].detach().cpu().numpy()
                output_dict['target_location'] = self._target_end_pos[:,:2].detach().cpu().numpy()
                #data = self.motion_agents[:self.T] 
                np.savez_compressed("saved_states.npz", **output_dict)
                print("Motion saved")
            self.t += 1
    
        if self.log_metrics == True:
            self.steps[env_ids] += 1

            Tmax = self.log_rim.size(1)
            n_idx = env_ids                                 

            # If steps started at 0 and incremented just above, the write index is step-1.
            t_idx = (self.steps[n_idx] - 1).clamp(min=0)     # (M,)
            t_idx = torch.minimum(t_idx, torch.tensor(Tmax-1, device=t_idx.device))

            # Batched write: one (env, t) per row
            self.log_rim[n_idx, t_idx, :, :] = self.obj_held_points[n_idx]
            self.d_last = d
        

        return obs.view(num_envs, num_agents, obs.shape[-1])

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = sum(self.task_obs_dim)
        return obs_size

    def _compute_reset(self):

        hand_positions_all = self._rigid_body_pos_tensor[:, :, self._lift_body_ids, :] # (n_envs, n_agents, 2, 3)
        contact_forces_all = self._contact_forces_tensor # (n_envs, n_agents, 15, 3)
        rigid_body_pos_all = self._rigid_body_pos_tensor # (n_envs, n_agents, 15, 3)

        hand_positions_all = hand_positions_all.view(-1, 2, 3)
        contact_forces_all = contact_forces_all.reshape(-1, 15, 3)
        rigid_body_pos_all = rigid_body_pos_all.reshape(-1, 15, 3)
        progress_buf_n_agents = self.progress_buf.clone().repeat_interleave(self.num_agents)
        dummy_shape = torch.ones(self.num_agents * self.num_envs, device=self.device, dtype=torch.long)

        obj_pos = self._obj_states[..., :3]
        obj_terminate_h = self.obj_height-0.12

        reset, terminate = compute_humanoid_reset(
            dummy_shape, progress_buf_n_agents, contact_forces_all,
            self._contact_body_ids, rigid_body_pos_all, self.max_episode_length,
            self._enable_early_termination, self._termination_heights, obj_pos, obj_height_termination = obj_terminate_h, num_humanoids=self.num_agents)
        
        self.reset_buf[:], self._terminate_buf[:] = reset, terminate

        return

    def _compute_reward(self, actions):
        w = self.w

        # Extract relevant tensors
        dt_tensor = torch.tensor(self.dt, dtype=torch.float32)
        stacked_root = self._humanoid_root_states_tensor

        root_pos = stacked_root[:,:,:3] # (n_envs, n_agents, 3)
        root_rot = stacked_root[:,:,3:7] # (n_envs, n_agents, 4)
        prev_root_pos = self._prev_root_pos_tensor # (n_envs, n_agents, 3)
        n_envs, n_agents, _ = prev_root_pos.shape
        obj_center = self._obj_states[..., 0:3]
        hands_pos = self._rigid_body_pos_tensor[:, :, self._lift_body_ids, :]
        rim_pts = self.obj_held_points   # (N_envs, K, 3)
        obj_rot = self._obj_states[..., 3:7]
        normals2d = self.normals2d

        # Organize tensors for reward calculations
        root_pos_all, root_rot_all, prev_root_pos_all, obj_center_all, \
            hands_pos_all, rim_pts_all, normals2d_all = prepare_tensors(root_pos, root_rot, prev_root_pos, \
                                                                        obj_center, hands_pos, rim_pts, obj_rot, normals2d, self.num_agents, self.K_RIM)

        # walk reward: 0,1,2
        r_pos, r_vel, r_face, self.dist_to_nearest_point, root_vel = compute_walk_reward(root_pos_all, root_rot_all, \
                                                                           prev_root_pos_all, obj_center_all, rim_pts_all, normals2d_all, dt_tensor,\
                                                                              standing_gap=0.3, dist_threshold=self.dist_threshold)
                
        # formation reward: 3
        ones = torch.ones(root_pos_all.shape[0], device=root_pos_all.device, dtype=root_pos_all.dtype)
        zeros = torch.zeros_like(ones)

        if w[3] == 0:
            r_form = r_cov = r_ang = zeros
        elif n_agents <= 1:
            r_form = r_cov = r_ang = ones
        else:
            r_ang = compute_angle_reward(root_pos_all, obj_center_all, self.mask, n_agents)
            r_cov = compute_coverage_reward(root_pos_all, rim_pts_all, self.mask, \
                                            self.obj_held_points_local, self.span_xy, n_agents)
            r_form = 0.75 * r_cov + 0.25 * r_ang # in retrospect, (0.5 * r_cov + 0.5 r_ang) might be better

        # hands and contacts reward: 4,5,6
        r_hand, r_contact, r_lift, contact_mask, hands_dist = compute_hands_and_lifts_reward(hands_pos_all, rim_pts_all, self.dist_to_nearest_point, \
                                                                                             dist_threshold=self.dist_threshold, held_target_z=self.target_height)
        
        # transport reward: 7,8
        target_pos = self._target_end_pos
        valid_humans = stacked_root[:,:,2] < 2
        two_hands_touch = contact_mask.sum(dim=-1) == 2
        two_hands_touch_per_env = two_hands_touch.view(-1, self.num_agents)
        move_mask = two_hands_touch_per_env.sum(-1) == valid_humans.sum(-1)

        if w[7] != 0:
            r_transport = compute_transport_reward(obj_center, target_pos, move_mask)
            r_transport = r_transport.unsqueeze(1).expand(-1, self.num_agents).reshape(-1)
            move_mask_all = torch.repeat_interleave(move_mask, repeats=self.num_agents, dim=0) 
            r_pos[move_mask_all] = 1.0
        else:
            r_transport = zeros
        
        if w[8] != 0:
            r_align = compute_align_reward(root_pos_all, root_rot_all, valid_humans, obj_center, target_pos, move_mask, n_agents)
            r_align = r_align.unsqueeze(1).expand(-1, self.num_agents).reshape(-1)
        else:
            r_align = zeros

        # putdown reward
        if w[9] != 0:
            r_putdown, reached = compute_putdown_reward(self.putdown_mask, self.num_agents, hands_pos_all, hands_dist, root_vel)

            # --- lock other rewards to 1 when reached ---
            for r in (r_hand, r_contact, r_transport, r_lift, r_align, r_form, r_ang, r_cov, r_face):
                r.masked_fill_(reached, 1.0)
        else:
            r_putdown = zeros


        total_reward = w[0] * r_pos + w[1] * r_vel + w[2] * (r_face * r_ang)**0.5 + \
        w[3] * r_form + w[4] * (r_hand * r_cov) + w[5] * r_contact + w[6] * (r_lift * r_cov) + \
        w[7] * r_transport + w[8] * r_align + w[9] * r_putdown

        self.rew_buf = total_reward.view(n_envs, n_agents)

        if self.log_metrics == True:
            ### start_t for metrics, when object starts moving
            hands_dist_reshape = hands_dist.view(-1, self.num_agents, 2)
            any_close = (hands_dist_reshape < 0.16).any(dim=2)
            any_close_env = any_close.any(dim=1)
            obj_lift = obj_center[:,2] > 0.86
            shift = obj_center[:,:2].norm(dim=-1) > 0.2 
            start_lift = any_close_env & obj_lift & shift
            self.start_lift |= start_lift
            self.start_t += (~self.start_lift).to(self.start_t.dtype)
            self.end_t += (~self.putdown_mask).to(self.end_t.dtype)

            # t_coop
            hands_near = (hands_dist < 0.12).to(hands_pos.dtype)  
            two_hands_near = hands_near.sum(dim=-1) == 2
            two_hands_near_per_env = two_hands_near.view(-1, self.num_agents)
            hands_near_mask = two_hands_near_per_env.sum(-1) == valid_humans.sum(-1)

            forces = self._contact_forces_tensor[:, :, [4, 5, 7, 8], :]  # (N, H, 4, 3)
            magnitudes = torch.norm(forces, dim=-1)  # (N, H, 4)
            nonzero_touch = magnitudes > 0  # (N, H, 4)
            any_contact_per_t = nonzero_touch.any(dim=-1)  # (N, H)
            arms_hands_contact_mask = any_contact_per_t.any(dim=-1)  # (N,)

            coop = arms_hands_contact_mask & hands_near_mask
            self.t_coop += coop.to(self.t_coop.dtype)
        
        return
    
    def _draw_task(self):

        # clear old markers
        self.gym.clear_lines(self.viewer)

        # reusable markers
        sphere_target_end = gymutil.WireframeSphereGeometry(0.1, 5, 5, None, color=(0, 1, 1))
        sphere_objhp  = gymutil.WireframeSphereGeometry(0.05, 8, 8, None, color=(0, 0, 1))

        pos_end = self._target_end_pos.detach().cpu().numpy()
        objhp = self.obj_held_points.detach().cpu().numpy()      # (N,K,3)

        pos_end[:,2] = 0
        
        for i, env_ptr in enumerate(self.envs):
            
            for k in range(objhp.shape[1]):
                p = objhp[i, k]
                pose_hp = gymapi.Transform(gymapi.Vec3(float(p[0]), float(p[1]), float(p[2])))
                gymutil.draw_lines(sphere_objhp, self.gym, self.viewer, env_ptr, pose_hp)
            
            # draw target marker
            pose_end = gymapi.Transform(gymapi.Vec3(float(pos_end[i,0]), float(pos_end[i,1]), float(pos_end[i,2])))
            gymutil.draw_lines(sphere_target_end, self.gym, self.viewer, env_ptr, pose_end)
            draw_circle(self.gym, self.viewer, env_ptr, pos_end[i], radius = 1)

        
def compute_obs(root_states, obj_pos, obj_held_pts, obs_vec6, tar_pos, nn_pts_z, putdown_mask, num_humanoids):

    # Root states
    ######################
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)  # (num_envs, 4)
    ######################

    #### pairwise rotation
    heading_rot_w = torch_utils.calc_heading_quat(root_rot)
    heading_rot_w_reshape = heading_rot_w.view(-1, num_humanoids, 4)
    heading_rot_w2l_reshape = heading_rot.view(-1, num_humanoids, 4)

    E, A, _ = heading_rot_w_reshape.shape
    q_w_j   = heading_rot_w_reshape.unsqueeze(1).expand(-1, A, -1, -1)   # (E, A, A, 4), j varies on dim=2
    q_w2l_i = heading_rot_w2l_reshape.unsqueeze(2).expand(-1, -1, A, -1) # (E, A, A, 4), i varies on dim=1
    pair_rot = quat_mul(q_w2l_i.reshape(-1, 4), q_w_j.reshape(-1, 4))
    pair_rot = pair_rot / pair_rot.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    pair_rot_tan_norm = torch_utils.quat_to_tan_norm(pair_rot).reshape(E,A,A,6)
    pair_rot_obs = pair_rot_tan_norm.view(E*A, A, 6)

    # Object observation
    ######################
    obj_pos[..., 2] = 1

    local_obj_pos = obj_pos - root_pos
    local_obj_pos_obs = quat_rotate(heading_rot, local_obj_pos)

    # Target observation
    ######################
    obj_pos[..., 2] = 0
    local_tar_pos = tar_pos - obj_pos
    local_tar_pos[:,2] = local_tar_pos[:,2] - nn_pts_z
    putdown_mask_all = torch.repeat_interleave(putdown_mask, repeats=num_humanoids, dim=0)    

    local_tar_pos[:,2][~putdown_mask_all] = 0
    local_tar_pos_obs = quat_rotate(heading_rot, local_tar_pos)
    
    # Object rim observation
    ######################
    ## obj_held_pts: (B, K, 3) in world frame
    B = root_pos.shape[0]
    K = obj_held_pts.shape[1]

    ## 1) Decide indexing in GLOBAL frame: nearest rim point to the root position
    d2 = (obj_held_pts - root_pos.unsqueeze(1)).pow(2).sum(dim=-1)  # (B,K), world-frame distances
    idx0 = d2.argmin(dim=-1)                                        # (B,)

    base = torch.arange(K, device=obj_held_pts.device).unsqueeze(0).expand(B, -1)  # (B,K)
    rolled = (base + idx0.unsqueeze(1)) % K                                      # (B,K)

    ## 2) Gather rolled points in WORLD frame
    rolled_world = obj_held_pts.gather(1, rolled.unsqueeze(-1).expand(-1, -1, 3))  # (B,K,3)

    ## 3) Convert to LOCAL (heading) frame after rolling
    rolled_local = quat_rotate_dimflex(
        heading_rot,
        rolled_world - root_pos.unsqueeze(1)                                       # (B,K,3)
    )   
    rim_obs = rolled_local.reshape(B, K * 3)
    ######################

    # Add relative rotations
    ######################
    other_agents_obs = pairwise_rotation_pos_obs(root_pos, obj_pos, heading_rot, num_humanoids)
    other_agents_obs = torch.cat([other_agents_obs, pair_rot_obs], dim=-1)
    other_agents_obs = other_agents_obs.view(other_agents_obs.shape[0], -1)

    obs = torch.cat([local_obj_pos_obs, rim_obs, obs_vec6, local_tar_pos_obs, other_agents_obs], dim=-1)
    ######################

    return obs


def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    contact_body_ids,
    rigid_body_pos,
    max_episode_length,
    enable_early_termination,
    termination_heights,
    obj_pos,
    obj_height_termination,
    num_humanoids,
):

    terminated = torch.zeros_like(reset_buf)

    # Early termination logic based on contact forces and body positions
    if enable_early_termination:
        # Mask the contact forces of the lifting body parts so they're not considered
        fall_masked_contact_buf = contact_buf.clone()
        fall_masked_contact_buf[:, contact_body_ids, :] = 0

        # Check if any body parts are making contact with a force above a minimal threshold
        # to determine if a fall contact has occurred.
        fall_contact = torch.any(torch.abs(fall_masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        # Check if the body height of any body parts is below a certain threshold
        # to determine if a fall due to height has occurred.
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        # Do not consider lifting body parts for the height check
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        # Combine the conditions to determine if the humanoid has fallen
        has_failed = torch.logical_and(fall_contact, fall_height)


        # ---------------- object height check --------------------
        obj_height = obj_pos[..., 2]             # (B,)
        obj_low = obj_height < obj_height_termination
        obj_high = obj_height > 1.30
        # --------------------------------------------------------------

        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1,
                        torch.ones_like(reset_buf), terminated)
    
    reset = reset.view(-1, num_humanoids).sum(-1).clamp(max = 1).bool()
    terminated = terminated.view(-1, num_humanoids).sum(-1).clamp(max = 1).bool()

    reset = (reset | obj_low | obj_high).long()
    terminated = (terminated | obj_low | obj_high).long()

    return reset, terminated



