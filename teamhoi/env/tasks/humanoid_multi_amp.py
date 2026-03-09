from enum import Enum
import numpy as np
import torch

from env.tasks.humanoid_multi import HumanoidMulti, dof_to_obs
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils

class HumanoidAMPMulti(HumanoidMulti):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAMPMulti.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._state_reset_happened = False

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        motion_file2 = cfg['env']['motion_file2']
        self._load_motion2(motion_file2) # for Masked AMP

        self._amp_obs_buf = torch.zeros((self.num_envs, self.num_agents, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float) # (N_envs, N_agents, 10, 125)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, :, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, :, 1:]
        
        self._amp_obs_demo_buf = None
        self._amp_obs_demo_buf2 = None

        self._amp_cond_buf = torch.zeros((self.num_envs, self.num_agents, self._num_amp_obs_steps, 1), device=self.device, dtype=torch.float32)
        self._curr_amp_cond_buf = self._amp_cond_buf[:, :, 0]
        self._hist_amp_cond_buf = self._amp_cond_buf[:, :, 1:]

        return

    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._update_hist_amp_cond()
        self._compute_amp_observations_batch()

        amp_obs_reshaped = self._amp_obs_buf.view(self.num_envs, self.num_agents, self.get_num_amp_obs()) # (N_envs * N_agents, 1250)
        self.extras["amp_obs"] = amp_obs_reshaped

        amp_cond_reshaped = self._amp_cond_buf.view(self.num_envs, self.num_agents, self._num_amp_obs_steps) # (N_envs * N_agents, 1250)
        self.extras["amp_cond"] = amp_cond_reshaped

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = self._motion_lib.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat
    
    def fetch_amp_obs_demo2(self, num_samples):

        if (self._amp_obs_demo_buf2 is None):
            self._build_amp_obs_demo_buf2(num_samples)
        else:
            assert(self._amp_obs_demo_buf2.shape[0] == num_samples)
        
        motion_ids = self._motion_lib2.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib2.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo2(motion_ids, motion_times0)
        self._amp_obs_demo_buf2[:] = amp_obs_demo.view(self._amp_obs_demo_buf2.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf2.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo
    

    def build_amp_obs_demo2(self, motion_ids, motion_times0):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib2.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
    
    def _build_amp_obs_demo_buf2(self, num_samples):
        self._amp_obs_demo_buf2 = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # TODO: Experimental # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return

    def _load_motion2(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib2 = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        if len(env_ids) > 0:
            self._state_reset_happened = True

        super()._reset_envs(env_ids)
        self._init_amp_obs(env_ids)

        return

    def _reset_actors(self, env_ids):

        self._humanoid_root_states_tensor[env_ids] = self._initial_humanoid_root_states_tensor[env_ids]

        # Sample x-y and rotations
        init_xy = torch_utils.sample_polar_no_collision_batch(B=len(env_ids), N=self.num_agents, R=8.0, R_max=8, d_min=0.4, \
                                       max_trials=100, oversample_factor=2, device=self._initial_humanoid_root_states_tensor.device)
                
        init_quaternion = torch_utils.sample_yaw_quaternion_batch(B=len(env_ids), N=self.num_agents)
        
        self._humanoid_root_states_tensor[env_ids, :, :2] = init_xy
        self._humanoid_root_states_tensor[env_ids, :, 3:7] = init_quaternion

        # Sample z to send humans to sky
        if self.fix_num_agents == 0:
            sample_mask = torch_utils.random_boolean_mask_uniform(len(env_ids), self.num_agents, N_min_agents=self.min_agents , device = self.device)
            #sample_mask = torch_utils.random_boolean_mask_weighted(len(env_ids), self.num_agents, weights= [0, 1, 0, 1], device = self.device)
        else:
            sample_mask = torch.ones(len(env_ids), self.fix_num_agents, device=self.device, dtype=torch.bool)
       
        self.mask[env_ids] = sample_mask

        if self.is_train == True:
            z = torch.ones_like(sample_mask, dtype = torch.float32) * 5 + 1
        else:
            z = torch.ones_like(sample_mask, dtype = torch.float32) * 5 + 10000
        z[sample_mask] = 0.89
        self._humanoid_root_states_tensor[env_ids, :, 2] = z

        self._dof_pos_tensor[env_ids] = torch.zeros_like(self._dof_pos_tensor[env_ids], device=self._dof_pos_tensor.device, dtype=torch.float)
        self._dof_vel_tensor[env_ids] = torch.zeros_like(self._dof_vel_tensor[env_ids], device=self._dof_vel_tensor.device, dtype=torch.float)
        self._reset_default_env_ids = env_ids
        return
    
    def _reset_default(self, env_ids):

        for i in range(self.num_agents):
            self._humanoid_root_states_list[i][env_ids] = self._initial_humanoid_root_states_list[i][env_ids]
            self._dof_pos_list[i][env_ids] = self._initial_dof_pos_list[i][env_ids]
            self._dof_vel_list[i][env_ids] = self._initial_dof_vel_list[i][env_ids]

        self._reset_default_env_ids = env_ids

        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == HumanoidAMPMulti.StateInit.Random
            or self._state_init == HumanoidAMPMulti.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidAMPMulti.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations_batch(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs

    
        curr_amp_cond = self._curr_amp_cond_buf[env_ids].unsqueeze(-2)
        self._hist_amp_cond_buf[env_ids] = curr_amp_cond

        return

    def _init_amp_obs_ref___(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _set_env_state_(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return
    
    def _set_env_state(
        self, env_ids, root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list,
        rigid_body_pos_list=None,
        rigid_body_rot_list=None,
        rigid_body_vel_list=None,
        rigid_body_ang_vel_list=None,
    ):
        for i in range(self.num_agents):

            self._humanoid_root_states_tensor[env_ids, i, 0:3] = root_pos_list[i]
            self._humanoid_root_states_tensor[env_ids, i, 3:7] = root_rot_list[i]
            self._humanoid_root_states_tensor[env_ids, i, 7:10] = root_vel_list[i]
            self._humanoid_root_states_tensor[env_ids, i, 10:13] = root_ang_vel_list[i]

            self._dof_pos_tensor[env_ids, i] = dof_pos_list[i]
            self._dof_vel_tensor[env_ids, i] = dof_vel_list[i]

        return
    
    def _refresh_sim_tensors(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        return


    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[2] - 1)):
                self._amp_obs_buf[:, :, i + 1] = self._amp_obs_buf[:, :, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[2] - 1)): # env_ids is never called
                self._amp_obs_buf[env_ids, :, i + 1] = self._amp_obs_buf[env_ids, :, i]
        return
    
    def _update_hist_amp_cond(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_cond_buf.shape[2] - 1)):
                self._amp_cond_buf[:, :, i + 1] = self._amp_cond_buf[:, :, i]
        else:
            for i in reversed(range(self._amp_cond_buf.shape[2] - 1)): # env_ids is never called
                self._amp_cond_buf[env_ids, :, i + 1] = self._amp_cond_buf[env_ids, :, i]
        
        return
    

    def _compute_amp_observations_batch(self, env_ids=None):
        num_envs = self.num_envs
        device = self._dof_pos_tensor.device

        # Default: use all environments
        if env_ids is None:
            env_ids = torch.arange(num_envs, device=device)

        if len(env_ids) == 0:
            return
        
        amp_obs = torch.zeros(len(env_ids) * self.num_agents, self._num_amp_obs_per_step, device = self.device) #TODO: hardcode

        rigid_body_pos = self._rigid_body_pos_tensor[:,:,0,:][env_ids] # (N_env, N_agents, 3)
        rigid_body_rot = self._rigid_body_rot_tensor[:,:,0,:][env_ids] # (N_env, N_agents, 4)
        rigid_body_vel = self._rigid_body_vel_tensor[:,:,0,:][env_ids] # (N_env, N_agents, 3)
        rigid_body_ang_vel = self._rigid_body_ang_vel_tensor[:,:,0,:][env_ids]  # (N_env, N_agents, 3)

        dof_pos = self._dof_pos_tensor[env_ids] # (N_env, N_agents, 28)
        dof_vel = self._dof_vel_tensor[env_ids] # (N_env, N_agents, 28)
        key_body_pos = self._rigid_body_pos_tensor[:,:,self._key_body_ids,:][env_ids] # (N_env, N_agents, 4, 3)

        n_envs, n_agents, _ = rigid_body_pos.shape

        mask = self.mask[env_ids].view(-1)

        rigid_body_pos_batch = rigid_body_pos.view(-1, 3)[mask]
        rigid_body_rot_batch = rigid_body_rot.view(-1, 4)[mask]
        rigid_body_vel_batch = rigid_body_vel.view(-1, 3)[mask]
        rigid_body_ang_vel_batch = rigid_body_ang_vel.view(-1, 3)[mask]
        dof_pos_batch = dof_pos.view(-1, 28)[mask]
        dof_vel_batch = dof_vel.view(-1, 28)[mask]
        key_body_pos_batch = key_body_pos.view(-1, 4, 3)[mask]

        # Build AMP observations in batch
        amp_obs_valid = build_amp_observations(
            rigid_body_pos_batch,
            rigid_body_rot_batch,
            rigid_body_vel_batch,
            rigid_body_ang_vel_batch,
            dof_pos_batch,
            dof_vel_batch,
            key_body_pos_batch,
            self._local_root_obs,
            self._root_height_obs,
            self._dof_obs_size,
            self._dof_offsets
        )

        ### Update AMP obs for post_physics_step and self.extras["amp_obs"]
        amp_obs[mask] = amp_obs_valid
        amp_obs = amp_obs.view(n_envs, n_agents, -1)
        self._curr_amp_obs_buf[env_ids] = amp_obs.clamp(min=-10, max = 10)

        ### Update Masked AMP condition for post_physics_step and self.extras["amp_cond"]
        dist_to_nearest_point = self.dist_to_nearest_point.view(num_envs, n_agents)
        dist_to_nearest_point[~self.mask] = 0
        self._curr_amp_cond_buf[env_ids] = dist_to_nearest_point[env_ids].unsqueeze(-1)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    
    """
    AI generated.
    Builds AMP (Adversarial Motion Prior) observations in a local heading-aligned frame.

    Generates features from the humanoid's root state, joint states, and key body positions, transformed into a local
    reference frame based on the root's heading. Includes optional root height and local root rotation representations.
    Used as input for AMP discriminators to evaluate motion realism.

    Args:
        root_pos (Tensor): Root positions of shape (B, 3).
        root_rot (Tensor): Root orientations as quaternions, shape (B, 4).
        root_vel (Tensor): Root linear velocities, shape (B, 3).
        root_ang_vel (Tensor): Root angular velocities, shape (B, 3).
        dof_pos (Tensor): DOF positions, shape (B, num_dofs).
        dof_vel (Tensor): DOF velocities, shape (B, num_dofs).
        key_body_pos (Tensor): Positions of key body parts, shape (B, K, 3).
        local_root_obs (bool): Whether to use local root orientation.
        root_height_obs (bool): Whether to include root height.
        dof_obs_size (int): Total observation size for DOFs.
        dof_offsets (List[int]): Offsets to select specific DOFs for observation.

    Returns:
        Tensor: Flattened AMP observation tensor of shape (B, D).
    """
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)

    return obs
