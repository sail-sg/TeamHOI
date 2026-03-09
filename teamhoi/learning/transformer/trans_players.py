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

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import learning.common_player as common_player
from utils import torch_utils


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

class TransPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._disc_reward_scale = config['disc_reward_scale']
        
        super().__init__(config)
        self.self_obs_size = sum(self.env.task.self_obs_dim)
        return

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        self._amp_masked_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])

        return
    
    
    def _build_net(self, config):
        super()._build_net(config)
        
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_masked_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)

            self._amp_input_mean_std.eval()
            self._amp_masked_input_mean_std.eval()  
        
        return

    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape

            if self.env.task._enable_task_obs:
                config['self_obs_size'] = self.env.task.get_obs_size() - self.env.task.get_task_obs_size()
                config['task_obs_size'] = self.env.task.get_task_obs_size()

                config['goal_obs_size'] = self.env.task.task_obs_dim[:-1]
                config['others_obs_each_size'] = self.env.task.task_obs_dim[-1] // self.env.task.num_agents
                config['num_agents'] = self.env.task.num_agents
                config['system_max_agents'] = self.env.task.system_max_agents
                config['goal_multiplier'] = self.env.task.goal_multiplier
            
            config["device"] = self.device
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
        return config

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            amp_cond = info['amp_cond']
            amp_cond = amp_cond[0:1]

            disc_pred = self._eval_disc(amp_obs)
            
            dist_threshold = 2.0
            sharpness = 15
            amp_blend_weight = torch.sigmoid((dist_threshold - amp_cond) * sharpness).mean(-1).unsqueeze(-1)
            amp_rewards = self._calc_amp_rewards(amp_obs, amp_blend_weight)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]

        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _preproc_amp_obs_masked(self, amp_obs):

        if amp_obs.ndim == 2:
            B,D = amp_obs.shape
            # mask arms and hands
            amp_obs = amp_obs.view(B, D//125, 125)
            amp_obs[:,:,25:49] = 0
            amp_obs[:,:,91:99] = 0
            amp_obs[:,:,113:119] = 0
            amp_obs = amp_obs.view(B, -1)
        elif amp_obs.ndim == 3:
            H, B, D = amp_obs.shape
            # mask arms and hands
            amp_obs = amp_obs.view(H, B, D//125, 125)
            amp_obs[:, :,:,25:49] = 0
            amp_obs[:, :,:,91:99] = 0
            amp_obs[:, :,:,113:119] = 0
            amp_obs = amp_obs.view(H, B, -1)

        if self._normalize_amp_input:
            amp_obs = self._amp_masked_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)
    
    def _eval_disc_masked(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs_masked(amp_obs)
        return self.model.a2c_network.eval_disc_masked(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs, amp_blend_w):
        disc_r = self._calc_disc_rewards(amp_obs, amp_blend_w)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs, amp_blend_w):
        with torch.no_grad():
            amp_obs_clone = amp_obs.clone()
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale

            disc_logits_masked = self._eval_disc_masked(amp_obs_clone)
            prob_masked = 1 / (1 + torch.exp(-disc_logits_masked)) 
            disc_r_masked = -torch.log(torch.maximum(1 - prob_masked, torch.tensor(0.0001, device=self.device)))
            disc_r_masked *= self._disc_reward_scale

        disc_r_blend = disc_r * (1 - amp_blend_w) + disc_r_masked * amp_blend_w
        return disc_r_blend
    
    def get_action(self, obs_dict, is_determenistic = False):
        obs = obs_dict['obs']
        N_envs, N_agents, _ = obs.shape
        obs_reshaped = obs.view(N_envs*N_agents, -1)

        mask = obs_dict['mask']

        if N_agents > 1:
            mask_others = torch_utils.expand_mask_wrt_others(mask)
            mask_others = mask_others.view(-1, N_agents - 1)
        else:
            mask_others = None
        
        mask = mask.view(-1)
        temp = torch.zeros_like(obs_reshaped, device=self.device)        
        obs_valid = obs_reshaped[mask]

        if self.normalize_input:
            obs_valid[:,:self.self_obs_size] = self.running_mean_std(obs_valid[:,:self.self_obs_size])
        
        temp[mask] = obs_valid
        processed_obs = temp

        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.states,
            'mask': mask,
            'mask_others': mask_others
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)

        output = self.get_actionn(res_dict, is_determenistic)
        return output

    def get_actionn(self, res_dict, is_determenistic = False):
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action