from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv
from rl_games.common.experience import ExperienceBuffer

from isaacgym.torch_utils import *

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn

import learning.replay_buffer as replay_buffer
import learning.common_agent as common_agent

from utils import torch_utils

import wandb
import math


class TransAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)
            self._amp_masked_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        self.epoch = 0
        self.env_num_agents = self.vec_env.env.task.num_agents
        self.env_total_w = sum(self.vec_env.env.task.w)
        self.env_total_w = 1
        self.dist_threshold = float(config.get("dist_threshold", 10000.0))

        self.num_envs = self.vec_env.env.task.cfg['env']['numEnvs']
        self.episodelen = self.vec_env.env.task.cfg['env']['episodeLength']

        self.reset_tick   = 0
        self.reset_order  = torch.arange(self.num_envs, device=self.device)  # or randperm
        self.base         = self.num_envs // self.episodelen
        self.rem          = self.num_envs %  self.episodelen
        self. cycle_active = True
        self.reset_step = 0
        return

    def init_tensors(self):
        super().init_tensors()

        #batch_size = self.env_num_agents * self.num_actors
        batch_size = self.num_actors

        algo_info = {
            #'num_actors' : self.num_actors*self.env_num_agents,
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        self.current_rewards = torch.zeros(self.num_actors, self.env_num_agents, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(self.num_actors, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((self.num_actors,), dtype=torch.uint8, device=self.ppo_device)

        n_horizon, n_envs, d_obs = self.experience_buffer.tensor_dict['obses'].shape

        del self.experience_buffer.tensor_dict['actions']
        del self.experience_buffer.tensor_dict['mus']
        del self.experience_buffer.tensor_dict['sigmas']
        del self.experience_buffer.tensor_dict['obses']
        del self.experience_buffer.tensor_dict['values']
        del self.experience_buffer.tensor_dict['neglogpacs']
        del self.experience_buffer.tensor_dict['rewards']

        batch_shape = self.experience_buffer.obs_base_shape # (32, n_envs)
        self.experience_buffer.tensor_dict['actions'] = torch.zeros(batch_shape + (self.env_num_agents, self.actions_num,),
                                                                dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['mus'] = torch.zeros(batch_shape + (self.env_num_agents, self.actions_num,),
                                                                dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['sigmas'] = torch.zeros(batch_shape + (self.env_num_agents, self.actions_num,),
                                                                dtype=torch.float32, device=self.ppo_device)
        
        self.experience_buffer.tensor_dict['sigmas'] = torch.zeros(batch_shape + (self.env_num_agents, self.actions_num,),
                                                                dtype=torch.float32, device=self.ppo_device)
        
        self.experience_buffer.tensor_dict['obses'] = torch.zeros((n_horizon, n_envs, self.env_num_agents, d_obs),dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['values'] = torch.zeros((n_horizon, n_envs, self.env_num_agents, 1),dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['neglogpacs'] = torch.zeros((n_horizon, n_envs, self.env_num_agents, 1),dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['rewards'] = torch.zeros((n_horizon, n_envs, self.env_num_agents, 1),dtype=torch.float32, device=self.ppo_device)
   
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])     
        self.experience_buffer.tensor_dict['mask'] = torch.zeros(batch_shape + (self.env_num_agents,), dtype = torch.bool, device = self.ppo_device)  

        self._build_amp_buffers()
        self.tensor_list += ['mask']

        return
    
    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()
            self._amp_masked_input_mean_std.eval()
        return

    def set_train(self):
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()
            self._amp_masked_input_mean_std.train()
        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state['amp_input_mean_std'] = self._amp_input_mean_std.state_dict()
            state['amp_masked_input_mean_std'] = self._amp_masked_input_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])
            self._amp_masked_input_mean_std.load_state_dict(weights['amp_masked_input_mean_std'])
        
        return

    def play_steps(self):
        self.set_eval()
        self.epoch += 1

        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):

            self.obs = self.env_reset(done_indices) # (N_envs * N_agents, 298)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                N_envs, N_agents, D = self.experience_buffer.tensor_dict[k][0].shape
                
                maskk = self.obs['mask'].view(-1)
                temp = torch.zeros((N_envs * N_agents, D), device = self.device)

                data = res_dict[k]
                if data.ndim == 1:
                    data = data.unsqueeze(-1)  # convert (N,) to (N, 1)

                temp[maskk] = data
                res_dict[k] = temp.view(N_envs, N_agents, -1)

                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            actions_reshaped = res_dict['actions'].view(-1, res_dict['actions'].shape[2])
            self.obs, rewards, self.dones, infos = self.env_step(actions_reshaped)
            
            # Gradually force environment resets to maintain balanced episode cycling during continued training
            if self.cycle_active:
                quota = (math.ceil((self.reset_step + 1) * self.num_envs / self.episodelen) - math.ceil(self.reset_step * self.num_envs / self.episodelen))

                if quota > 0:
                    inds = self.reset_order[self.reset_tick: self.reset_tick + quota]
                    force_mask = torch.zeros_like(self.dones)
                    force_mask[inds] = 1
                    self.dones = torch.maximum(self.dones, force_mask)

                    self.reset_tick += quota
                self.reset_step += 1
                
                if self.reset_tick >= self.num_envs:   # one full pass done
                    self.cycle_active = False


            rewards = rewards.permute(0,2,1)
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])
            self.experience_buffer.update_data('mask', n, infos['mask'])

            self.experience_buffer.update_data('amp_cond', n, infos['amp_cond'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs, infos['mask'])

            temp = torch.zeros((N_envs * N_agents, 1), device = self.device)
            temp[maskk] = next_vals
            next_vals = temp

            next_vals = next_vals.view(self.num_actors, self.env_num_agents, -1)
            next_vals *= (1.0 - terminated).unsqueeze(-1)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards.squeeze(-1)
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)
                
            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards_task = self.experience_buffer.tensor_dict['rewards']
        mb_rewards_task_normalized = mb_rewards_task.clone() * 1 / self.env_total_w

        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']

        mb_amp_cond = self.experience_buffer.tensor_dict['amp_cond']

        H, E, A, _ = mb_amp_obs.shape
        mb_amp_obs = mb_amp_obs.view(H, E*A, -1)
        mb_amp_cond = mb_amp_cond.view(H, E*A, -1)

        dist_threshold = self.dist_threshold
        sharpness = 15

        amp_blend_weight = torch.sigmoid((dist_threshold - mb_amp_cond) * sharpness).mean(-1).unsqueeze(-1) # inside (masked) -> 1.0, full (outside) -> 0.

        amp_rewards = self._calc_amp_rewards(mb_amp_obs, amp_blend_weight)
        amp_rewards['disc_rewards'] = amp_rewards['disc_rewards'].view(H, E, A, -1)
        mb_rewards = self._combine_rewards(mb_rewards_task, amp_rewards)
        mb_rewards_normalized = self._combine_rewards(mb_rewards_task_normalized, amp_rewards)

        mask = self.experience_buffer.tensor_dict['mask']

        wandb.log({"Rewards/task_rewards": torch.mean(mb_rewards_task_normalized[mask]),
            "Rewards/amp_rewards": torch.mean(amp_rewards['disc_rewards'][mask]),
            "Rewards/all_rewards": torch.mean(mb_rewards_normalized[mask]),
            "epoch": self.epoch})


        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns) # (32 * N_envs, N_agents, 1)
        batch_dict['played_frames'] = self.batch_size

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict
    
    def get_action_values(self, obs_dict, rand_action_probs= None):
        
        obs_reshaped = obs_dict['obs'].view(-1, obs_dict['obs'].shape[2])
        mask = obs_dict['mask']

        N_envs, N_agents = mask.shape
    
        if N_agents > 1:
            mask_others = torch_utils.expand_mask_wrt_others(mask)
            mask_others = mask_others.view(-1, N_agents - 1)
        else:
            mask_others = None
        
        mask = mask.view(-1)
        temp = torch.zeros_like(obs_reshaped, device=self.device)
        processed_obs = obs_reshaped[mask]

        if self.normalize_input:
            processed_obs[:,:self.self_obs_size] = self.running_mean_std(processed_obs[:,:self.self_obs_size])

        temp[mask] = processed_obs
        processed_obs = temp

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states,
            'mask': mask,
            'mask_others': mask_others
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs_dict['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)

        if rand_action_probs is not None:        
            rand_action_mask = torch.bernoulli(rand_action_probs)
            det_action_mask = rand_action_mask == 0.0
            res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]
            res_dict['rand_action_mask'] = rand_action_mask

        return res_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict['amp_obs'] = batch_dict['amp_obs']
        self.dataset.values_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        self.dataset.values_dict['amp_obs_demo2'] = batch_dict['amp_obs_demo2']
        self.dataset.values_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']
        
        self.dataset.values_dict['amp_cond'] = batch_dict['amp_cond']
        self.dataset.values_dict['amp_cond_replay'] = batch_dict['amp_cond_replay']
        
        rand_action_mask = batch_dict['rand_action_mask']
        self.dataset.values_dict['rand_action_mask'] = rand_action_mask
        self.dataset.values_dict['mask'] = batch_dict['mask']
        self.dataset.values_dict['mask_replay'] = batch_dict['mask_replay']
        return

    def train_epoch(self):
        #TODO:
        play_time_start = time.time()

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps() # (HorizonLength * N_envs * N_agents, X)

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)
        
        self._update_amp_demos()
        num_obs_samples = batch_dict['amp_obs'].shape[0] # (HorizonLength * N_envs , N_agents, X)

        amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)['amp_obs']
        batch_dict['amp_obs_demo'] = amp_obs_demo

        batch_dict['amp_obs_demo2'] = self._amp_obs_demo_buffer2.sample(num_obs_samples)['amp_obs']

        if (self._amp_replay_buffer.get_total_count() == 0):
            batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
            batch_dict['mask_replay'] = batch_dict['mask']
            batch_dict['amp_cond_replay'] = batch_dict['amp_cond']
        else:
            replay_dict = self._amp_replay_buffer.sample(num_obs_samples)
            batch_dict['amp_obs_replay'] = replay_dict['amp_obs']
            batch_dict['mask_replay'] = replay_dict['mask'].bool()
            batch_dict['amp_cond_replay'] = replay_dict['amp_cond']
        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])
                
                if self.schedule_type == 'legacy':  
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)
            
            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        self._store_replay_amp_obs(batch_dict['amp_obs'], batch_dict['mask'], batch_dict['amp_cond'])

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def calc_gradients(self, input_dict):

        self.set_train()

        def reshaper(x):
            N, M, _ = x.shape
            return x.view(N*M, -1)

        value_preds_batch = reshaper(input_dict['old_values'])
        old_action_log_probs_batch = reshaper(input_dict['old_logp_actions'])
        advantage = reshaper(input_dict['advantages'])
        old_mu_batch = reshaper(input_dict['mu'])
        old_sigma_batch = reshaper(input_dict['sigma'])
        return_batch = reshaper(input_dict['returns'])
        actions_batch = reshaper(input_dict['actions'])
        
        obs_batch = reshaper(input_dict['obs'])

        mask_obs = reshaper(input_dict['mask'].unsqueeze(-1)).squeeze(-1)

        temp = torch.zeros_like(obs_batch, device = self.device)

        obs_valid = obs_batch[mask_obs]

        if self.normalize_input:
            obs_valid[:,:self.self_obs_size] = self.running_mean_std(obs_valid[:,:self.self_obs_size])
        
        temp[mask_obs] = obs_valid
        obs_batch = temp

        value_preds_batch = value_preds_batch[mask_obs]
        old_action_log_probs_batch = old_action_log_probs_batch[mask_obs]
        advantage = advantage[mask_obs]
        old_mu_batch = old_mu_batch[mask_obs]
        old_sigma_batch = old_sigma_batch[mask_obs]
        return_batch = return_batch[mask_obs]
        actions_batch = actions_batch[mask_obs]
        
        amp_multiplier = 1.5

        mask_amp = reshaper(input_dict['mask'][0:int(self._amp_minibatch_size * amp_multiplier)].unsqueeze(-1)).squeeze(-1)
        amp_obs = reshaper(input_dict['amp_obs'][0:int(self._amp_minibatch_size * amp_multiplier)])


        amp_obs_clone = amp_obs[mask_amp].clone()
        amp_obs_masked = self._preproc_amp_obs_masked(amp_obs_clone)

        amp_cond = reshaper(input_dict['amp_cond'][0:int(self._amp_minibatch_size * amp_multiplier)])[mask_amp]
        outside = amp_cond.min(dim=-1)[0] > self.dist_threshold
        
        n_outside = outside.sum()
        if n_outside > 500:
            amp_obs = self._preproc_amp_obs(amp_obs[mask_amp][outside])
        else:
            amp_obs = None

        ## Take note of amp_obs_replay
        mask_amp_replay = reshaper(input_dict['mask_replay'][0:int(self._amp_minibatch_size * amp_multiplier)].unsqueeze(-1)).squeeze(-1)
        amp_obs_replay = reshaper(input_dict['amp_obs_replay'][0:int(self._amp_minibatch_size * amp_multiplier)])

        amp_obs_replay_clone = amp_obs_replay[mask_amp_replay].clone()
        amp_obs_replay_masked = self._preproc_amp_obs_masked(amp_obs_replay_clone)

        amp_cond_replay = reshaper(input_dict['amp_cond_replay'][0:int(self._amp_minibatch_size * amp_multiplier)])[mask_amp_replay]
        outside_replay = amp_cond_replay.min(dim=-1)[0] > self.dist_threshold

        if outside_replay.sum() > 500 and n_outside > 500:
            amp_obs_replay = self._preproc_amp_obs(amp_obs_replay[mask_amp_replay][outside_replay])
        else:
            amp_obs_replay = None

        # Demo 1 --> Full
        input_dict['amp_obs_demo'] = input_dict['amp_obs_demo'].unsqueeze(1).expand(-1, self.env_num_agents, -1)
        amp_obs_demo = input_dict['amp_obs_demo'][0:int(self._amp_minibatch_size)]
        amp_obs_demo = amp_obs_demo.reshape(-1, amp_obs_demo.shape[2])

        # Demo 2 -- > Masked
        input_dict['amp_obs_demo2'] = input_dict['amp_obs_demo2'].unsqueeze(1).expand(-1, self.env_num_agents, -1)
        amp_obs_demo2 = input_dict['amp_obs_demo2'][0:int(self._amp_minibatch_size)]
        amp_obs_demo2 = amp_obs_demo2.reshape(-1, amp_obs_demo2.shape[2])
        amp_obs_demo_clone2 = amp_obs_demo2.clone()
        amp_obs_demo_masked2 = self._preproc_amp_obs_masked(amp_obs_demo_clone2)
        amp_obs_demo_masked2.requires_grad_(True)

        if n_outside > 500:
            num_true = int(outside.sum().item())
            amp_obs_mask = torch.zeros(amp_obs_demo.shape[0], dtype=torch.bool, device=amp_obs_demo.device)

            # randomly select indices to set True
            true_indices = torch.randperm(amp_obs_demo.shape[0], device=amp_obs_demo.device)[:num_true]
            amp_obs_mask[true_indices] = True
            amp_obs_demo = self._preproc_amp_obs(amp_obs_demo[amp_obs_mask])
            amp_obs_demo.requires_grad_(True)
        else:
            amp_obs_demo = None
        
        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        B, N_agents = input_dict['mask'].shape
    
        if N_agents > 1:
            mask_others = torch_utils.expand_mask_wrt_others(input_dict['mask'])
            mask_others = mask_others.view(-1, N_agents - 1)
        else:
            mask_others = None
        
        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'amp_obs' : amp_obs,
            'amp_obs_replay' : amp_obs_replay,
            'amp_obs_demo' : amp_obs_demo,
            'amp_obs_demo2' : amp_obs_demo2,
            'mask': mask_obs,
            'mask_others': mask_others,
            'amp_cond': amp_cond,
            'amp_cond_replay': amp_cond_replay,
            'amp_obs_masked' : amp_obs_masked,
            'amp_obs_replay_masked' : amp_obs_replay_masked,
            'amp_obs_demo_masked2' : amp_obs_demo_masked2,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp'].unsqueeze(-1)
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            disc_agent_logit = res_dict['disc_agent_logit']
            disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
            disc_demo_logit = res_dict['disc_demo_logit']

            disc_agent_logit_masked = res_dict['disc_agent_logit_masked']
            disc_agent_replay_logit_masked = res_dict['disc_agent_replay_logit_masked']
            disc_demo_logit_masked = res_dict['disc_demo_logit_masked']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            a_clipped = a_info['actor_clipped'].float()

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)
            
            c_loss = torch.mean(c_loss)

            a_loss = torch.mean(a_loss)
            entropy = torch.mean(entropy) 
            b_loss = torch.mean(b_loss) 
            a_clip_frac = torch.mean(a_clipped) 

            if None in (disc_agent_logit, disc_agent_replay_logit, disc_demo_logit):
                disc_loss = torch.tensor(0.0, device=self.device)
            else:
                disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
                disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
                disc_loss = disc_info['disc_loss']


            disc_agent_cat_logit_masked = torch.cat([disc_agent_logit_masked, disc_agent_replay_logit_masked], dim=0)
            disc_info_masked = self._disc_loss_masked(disc_agent_cat_logit_masked, disc_demo_logit_masked, amp_obs_demo_masked2)
            disc_loss_masked = disc_info_masked['disc_loss']

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                 + self._disc_coef * disc_loss + self._disc_coef * disc_loss_masked
            
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

              
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        #self.train_result.update(a_info)
        #self.train_result.update(c_info)
        #self.train_result.update(disc_info)
        
        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        # when eps greedy is enabled, rollouts will be generated using a mixture of
        # a deterministic and stochastic actions. The deterministic actions help to
        # produce smoother, less noisy, motions that can be used to train a better
        # discriminator. If the discriminator is only trained with jittery motions
        # from noisy actions, it can learn to phone in on the jitteriness to
        # differential between real and fake samples.
        self._enable_eps_greedy = bool(config['enable_eps_greedy'])

        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        self._amp_observation_space = self.env_info['amp_observation_space']
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])
        assert(self._amp_minibatch_size <= self.minibatch_size)

        self._disc_coef = config['disc_coef']
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape

        if self.vec_env.env.task._enable_task_obs:
            config['self_obs_size'] = self.vec_env.env.task.get_obs_size() - self.vec_env.env.task.get_task_obs_size()
            config['task_obs_size'] = self.vec_env.env.task.get_task_obs_size()
            config['goal_obs_size'] = self.vec_env.env.task.task_obs_dim[:-1]
            config['others_obs_each_size'] = self.vec_env.env.task.task_obs_dim[-1] // self.vec_env.env.task.num_agents
            config['num_agents'] = self.vec_env.env.task.num_agents
            config['system_max_agents'] = self.vec_env.env.task.system_max_agents
            config['goal_multiplier'] = self.vec_env.env.task.goal_multiplier

        config["device"] = self.device

        return config
    
    def _build_rand_action_probs(self):
        num_envs = self.vec_env.env.task.num_envs
        env_ids = to_torch(np.arange(num_envs), dtype=torch.float32, device=self.ppo_device)

        self._rand_action_probs = 1.0 - torch.exp(10 * (env_ids / (num_envs - 1.0) - 1.0))
        self._rand_action_probs[0] = 1.0
        self._rand_action_probs[-1] = 0.0
        
        if not self._enable_eps_greedy:
            self._rand_action_probs[:] = 1.0

        return

    def _init_train(self):
        super()._init_train()
        self._init_amp_demo_buf()
        return
    
    def compute_disc_grad_penalty(self, obs_demo):
        # we need grads w.r.t. inputs:
        obs_demo = obs_demo.detach().requires_grad_(True)

        # force the SDPA "math" backend so double-backward is supported
        with torch.backends.cuda.sdp_kernel(enable_math=True,
                                            enable_flash=False,
                                            enable_mem_efficient=False):
            logits = self.model.a2c_network._disc_logits(self.model.a2c_network._eval_Transformer_disc(obs_demo))

        grad = torch.autograd.grad(
            outputs=logits,
            inputs=obs_demo,
            grad_outputs=torch.ones_like(logits),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gp = (grad.pow(2).sum(dim=-1)).mean()
        return gp

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty.detach(),
            'disc_logit_loss': disc_logit_loss.detach(),
            'disc_agent_acc': disc_agent_acc.detach(),
            'disc_demo_acc': disc_demo_acc.detach(),
            'disc_agent_logit': disc_agent_logit.detach(),
            'disc_demo_logit': disc_demo_logit.detach()
        }
        return disc_info
    

    def _disc_loss_masked(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights_masked()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights_masked()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty.detach(),
            'disc_logit_loss': disc_logit_loss.detach(),
            'disc_agent_acc': disc_agent_acc.detach(),
            'disc_demo_acc': disc_demo_acc.detach(),
            'disc_agent_logit': disc_agent_logit.detach(),
            'disc_demo_logit': disc_demo_logit.detach()
        }
        return disc_info

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo
    
    def _fetch_amp_obs_demo2(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo2(num_samples)
        return amp_obs_demo


    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['amp_obs'] = torch.zeros((batch_shape[0],batch_shape[1], self.env_num_agents, self._amp_observation_space.shape[0]),
                                                                    device=self.ppo_device)
        #self.experience_buffer.tensor_dict['rand_action_mask'] = torch.zeros(batch_shape, dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['rand_action_mask'] = torch.ones(batch_shape, dtype=torch.float32, device=self.ppo_device)
        
        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)

        self._amp_obs_demo_buffer2 = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)

        self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)
        
        self._build_rand_action_probs()

        self.experience_buffer.tensor_dict['amp_cond'] = torch.zeros((batch_shape[0],batch_shape[1], self.env_num_agents, self._amp_observation_space.shape[0] // 125),
                                                            device=self.ppo_device) # TODO: Hardcode
        
        self.tensor_list += ['amp_obs', 'rand_action_mask', 'amp_cond']
        return

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

            curr_samples = self._fetch_amp_obs_demo2(self._amp_batch_size)
            self._amp_obs_demo_buffer2.store({'amp_obs': curr_samples})

        return
    
    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})

        new_amp_obs_demo = self._fetch_amp_obs_demo2(self._amp_batch_size)
        self._amp_obs_demo_buffer2.store({'amp_obs': new_amp_obs_demo})
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs
    
    def _preproc_amp_obs_masked(self, amp_obs):
        if amp_obs.ndim == 2:
            B,D = amp_obs.shape
            amp_obs = amp_obs.view(B, D//125, 125)
            amp_obs[:,:,25:49] = 0
            amp_obs[:,:,91:99] = 0
            amp_obs[:,:,113:119] = 0
            amp_obs = amp_obs.view(B, -1)
        elif amp_obs.ndim == 3:
            H, B, D = amp_obs.shape
            amp_obs = amp_obs.view(H, B, D//125, 125)
            amp_obs[:, :,:,25:49] = 0
            amp_obs[:, :,:,91:99] = 0
            amp_obs[:, :,:,113:119] = 0
            amp_obs = amp_obs.view(H, B, -1)

        if self._normalize_amp_input:
            amp_obs = self._amp_masked_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']
        
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _eval_disc_masked(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs_masked(amp_obs)
        return self.model.a2c_network.eval_disc_masked(proc_amp_obs)
    
    def _eval_critic(self, obs_dict, mask):
        self.model.eval()
        obs = obs_dict['obs']

        obs = obs.view(-1, obs.shape[2])
        N_envs, N_agents = mask.shape
    
        if N_agents > 1:
            mask_others = torch_utils.expand_mask_wrt_others(mask)
            mask_others = mask_others.view(-1, N_agents - 1)
        else:
            mask_others = None
        
        mask = mask.view(-1)

        temp = torch.zeros_like(obs, device = self.device)
        obs_valid = obs[mask]

        if self.normalize_input:
            obs_valid[:,:self.self_obs_size] = self.running_mean_std(obs_valid[:,:self.self_obs_size])

        temp[mask] = obs_valid
        processed_obs = temp

        value = self.model.a2c_network.eval_critic(processed_obs, mask, mask_others)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _calc_advs_ori(self, batch_dict):
        returns = batch_dict['returns']
        values = batch_dict['values']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=-1)

        if self.normalize_advantage:
            advantages = (advantages - advantages[batch_dict['mask']].mean()) / (advantages[batch_dict['mask']].std() + 1e-8)

        return advantages.unsqueeze(-1)

    def _calc_advs(self, batch_dict):
        returns = batch_dict['returns']
        values = batch_dict['values']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=-1)

        if self.normalize_advantage:
            mask = batch_dict['mask'] 
            advantages[~mask] = 0
            group_ids = mask.sum(-1)  # (B,) with values in [0, N]
            num_groups = group_ids.max().item() + 1  # total group count

            # Compute per-group counts
            group_counts = torch.bincount(group_ids, minlength=num_groups).float()

            # Expand group_ids to match (B, N)
            expanded_group_ids = group_ids.unsqueeze(1).expand(-1, advantages.shape[1])  # (B, N)

            # Sum of advantages per group: (G, N)
            sum_adv = torch.zeros(num_groups, advantages.shape[1], device=advantages.device)
            sum_adv.scatter_add_(0, expanded_group_ids, advantages)
            sum_adv_row = sum_adv.sum(-1)

            #group_counts_n_humans = group_counts * torch.arange(mask.shape[1]+1, device = self.device)
            group_counts_n_humans = group_counts * torch.arange(num_groups, device = self.device)
            mean_adv_row = sum_adv_row / (group_counts_n_humans + 1e-8)

            # Mean per group
            advantages_mean = advantages - mean_adv_row[group_ids].unsqueeze(-1)

            # Calculate std
            advantages_mean_mask = advantages_mean.clone()
            advantages_mean_mask[~mask] = 0
            advantages_mean_mask_sq = advantages_mean_mask**2

            std_adv = torch.zeros(num_groups, advantages_mean_mask_sq.shape[1], device=self.device)
            std_adv.scatter_add_(0, expanded_group_ids, advantages_mean_mask_sq)
            std_adv_row = std_adv.sum(-1)
            std_adv_row = (std_adv_row / (group_counts_n_humans - 1))**0.5

            std = std_adv_row[group_ids] + 1e-8
            advantages = advantages_mean / std.unsqueeze(-1)

        return advantages.unsqueeze(-1)

    def _calc_amp_rewards(self, amp_obs, amp_blend_w):
        disc_r = self._calc_disc_rewards(amp_obs, amp_blend_w)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs, amp_blend_w):
        with torch.no_grad():
            amp_obs_clone = amp_obs.clone()

            disc_logits = self._eval_disc(amp_obs) # amp_obs -> horizonLength, n_envs*n_agents, 1250
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._disc_reward_scale

            disc_logits_masked = self._eval_disc_masked(amp_obs_clone)
            prob_masked = 1 / (1 + torch.exp(-disc_logits_masked)) 
            disc_r_masked = -torch.log(torch.maximum(1 - prob_masked, torch.tensor(0.0001, device=self.device)))
            disc_r_masked *= self._disc_reward_scale

        disc_r_blend = disc_r * (1 - amp_blend_w) + disc_r_masked * amp_blend_w
        return disc_r_blend

    def _store_replay_amp_obs(self, amp_obs, mask=None, amp_cond = None):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

            if mask is not None:
                mask = mask[keep_mask]
            
            if amp_cond is not None:
                amp_cond = amp_cond[keep_mask]

        if (amp_obs.shape[0] > buf_size):
            rand_idx = torch.randperm(amp_obs.shape[0])
            rand_idx = rand_idx[:buf_size]
            amp_obs = amp_obs[rand_idx]

            if mask is not None:
                mask = mask[rand_idx]

            if amp_cond is not None:
                amp_cond = amp_cond[rand_idx]

        self._amp_replay_buffer.store({'amp_obs': amp_obs, 'mask': mask, 'amp_cond': amp_cond})
        return

    
    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        self.writer.add_scalar('losses/disc_loss', torch_ext.mean_list(train_info['disc_loss']).item(), frame)

        self.writer.add_scalar('info/disc_agent_acc', torch_ext.mean_list(train_info['disc_agent_acc']).item(), frame)
        self.writer.add_scalar('info/disc_demo_acc', torch_ext.mean_list(train_info['disc_demo_acc']).item(), frame)
        
        self.writer.add_scalar('info/disc_agent_logit', torch.mean(torch.cat(train_info['disc_agent_logit'], dim=0)).item(), frame)

        self.writer.add_scalar('info/disc_demo_logit', torch_ext.mean_list(train_info['disc_demo_logit']).item(), frame)
        self.writer.add_scalar('info/disc_grad_penalty', torch_ext.mean_list(train_info['disc_grad_penalty']).item(), frame)
        self.writer.add_scalar('info/disc_logit_loss', torch_ext.mean_list(train_info['disc_logit_loss']).item(), frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)
        return

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1, 0]

            amp_cond = info['amp_cond']
            amp_cond = amp_cond[0:1, 0]

            disc_pred = self._eval_disc(amp_obs)

            dist_threshold = self.dist_threshold
            sharpness = 15
            amp_blend_weight = torch.sigmoid((dist_threshold - amp_cond) * sharpness).mean(-1).unsqueeze(-1)
            amp_rewards = self._calc_amp_rewards(amp_obs, amp_blend_weight)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)
        return
