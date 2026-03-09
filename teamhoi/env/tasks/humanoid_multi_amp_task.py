import torch

import env.tasks.humanoid_multi_amp as humanoid_amp_multi
import time

class HumanoidAMPTaskMulti(humanoid_amp_multi.HumanoidAMPMulti):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        return

    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        return

    def render(self, sync_frame_time=False):

        if self.viewer:
            self._draw_task()
        super().render(sync_frame_time)
        return
    
    def _update_task(self):
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        return

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        
        humanoid_obs_tensor = torch.zeros(len(env_ids) * self.num_agents, 223, device = self.device)
        humanoid_obs_tensor_valid, mask = self._compute_humanoid_obs_tensor(env_ids) # (N_envs, N_agents, 223)
        humanoid_obs_tensor[mask] = humanoid_obs_tensor_valid
        humanoid_obs_tensor = humanoid_obs_tensor.view(len(env_ids), self.num_agents, -1)
        
        if (self._enable_task_obs):
            task_obs_tensor = self._compute_task_obs_tensor(env_ids) # (N_envs, N_agents, 75)

        task_obs_dim = sum(self.task_obs_dim)
        N_envs, N_agents, fill_dim = task_obs_tensor.shape
        task_obs_init = torch.ones((N_envs, N_agents, task_obs_dim), device = task_obs_tensor.device) * -5.0

        task_obs_init[:,:,:fill_dim] = task_obs_tensor

        obs = torch.cat([humanoid_obs_tensor, task_obs_init], dim=-1)
        self.obs_buf[env_ids] = obs
        
        return

    def _compute_task_obs(self, env_ids=None):
        return NotImplemented
    
    def _compute_task_obs_tensor(self, env_ids=None):
        return NotImplemented

    def _compute_reward(self, actions):
        return NotImplemented

    def _draw_task(self):
        return