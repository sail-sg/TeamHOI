# ================================================================
# Tokenization + Attention + Padding
#
# Observations are split into self, object, goal, and other agents.
# Each part is encoded into tokens via MLPs.
#
# Transformer queries = [weight_token, self, object, goal]
# Other-agent tokens are used as key/value in cross-attention.
#
# The transformer alternates:
#   self-attention (task reasoning)
#   cross-attention (inter-agent reasoning)
#
# Number of agents can vary. We zero-pad other-agent observations
# to a fixed maximum size so tensor shapes stay consistent and the
# model generalizes across team sizes. It performs better in
# zero-shot settings than no padding version.
# ================================================================

from learning.amp_network_builder import AMPBuilder
from learning.transformer.composer import Composer

import torch
import torch.nn as nn

DISC_LOGIT_INIT_SCALE = 1.0

class AMPTransformerMultiTaskBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPTransformerMultiTaskBuilder.UnifiedNetworkClass(self.params, **kwargs)
        return net
    
    class UnifiedNetworkClass(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs["self_obs_size"]
            self.task_obs_size = kwargs["task_obs_size"]

            task_obs_size_list = kwargs['goal_obs_size']
            self.obj_obs_size = sum(task_obs_size_list[:-1])
            self.goal_obs_size = task_obs_size_list[-1]
            self.others_obs_each_size = kwargs["others_obs_each_size"]
            self.num_agents = kwargs["num_agents"]
            self.system_max_agents = kwargs["system_max_agents"]
        
            self.task_obs_each_size = [self.obj_obs_size, self.goal_obs_size, self.others_obs_each_size]

            super().__init__(params, **kwargs)

            del self.actor_mlp, self.mu, self.mu_act # delete useless networks
            
            self.device = kwargs["device"]
            self.dof_action_size = kwargs['actions_num']

            self._build_Transformer(params, **kwargs)
            
            del self.critic_mlp, self.critic_cnn, self.value, self.actor_cnn
            self._build_Transformer_critic(params, **kwargs)

            self.mult = kwargs["goal_multiplier"]

            return

        def _build_Transformer(self, params, **kwargs):

            num_features = params["transformer"]["num_features"]
            drop_ratio = 0.0 # using dropout will make training failed [BUG]
            tokenizer_units = params["transformer"]["tokenizer_units"]

            print("build tokenizer for self obs")
            mlp_args = {
                'input_size' : self.self_obs_size, 
                'units' : tokenizer_units + [num_features], 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.self_encoder = self._build_mlp(**mlp_args)

            self.task_encoder = nn.ModuleList()
            for idx, i in enumerate(self.task_obs_each_size):
                print("build tokenizer for subtask obs with size {}".format(i))
                mlp_args = {
                    'input_size' : i, 
                    'units' : tokenizer_units + [num_features], 
                    'activation' : self.activation, 
                    'norm_func_name' : self.normalization,
                    'dense_func' : torch.nn.Linear,
                    'd2rl' : self.is_d2rl,
                    'norm_only_first_layer' : self.norm_only_first_layer
                }
                self.task_encoder.append(self._build_mlp(**mlp_args))

            mlp_init = self.init_factory.create(**{"name": "default"})
            for nets in [self.self_encoder, self.task_encoder]:
                for m in nets.modules():
                    if isinstance(m, nn.Linear):
                        mlp_init(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)
            
            self.weight_token = nn.Parameter(torch.zeros(1, 1, num_features)) # extra token
            self.weight_padding = nn.Parameter(torch.zeros(1, 1, num_features))

            d_model = num_features
            self.transformer_encoder = AlternatingSelfCrossTransformer(
                d_model=d_model,
                nhead=params["transformer"]["layer_num_heads"],
                num_layers=params["transformer"]["num_layers"],       # e.g., 6 -> S, C, S, C, S, C
                dim_feedforward=params["transformer"]["layer_dim_feedforward"],
                dropout=drop_ratio,
                activation='relu',
                batch_first=True,
            )

            # weight init
            nn.init.trunc_normal_(self.weight_token, std=0.02)

            # extra mlp
            mlp_args = {
                'input_size': num_features,
                'units': params["transformer"]["extra_mlp_units"],
                'activation': self.activation,
                'dense_func': torch.nn.Linear,
            }

            output_size = kwargs['actions_num']
            comp_act = "identity"
            
            self.composer = Composer(mlp_args, output_size=output_size, activation=comp_act)

            return
        
        def _build_Transformer_critic(self, params, **kwargs):

            num_features = params["transformer"]["num_features"]
            drop_ratio = 0.0 # using dropout will make training failed [BUG]
            tokenizer_units = params["transformer"]["tokenizer_units"]

            print("build tokenizer for self obs")
            mlp_args = {
                'input_size' : self.self_obs_size, 
                'units' : tokenizer_units + [num_features], 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.self_encoder_critic = self._build_mlp(**mlp_args)

            self.task_encoder_critic = nn.ModuleList()
            for idx, i in enumerate(self.task_obs_each_size):
                print("build tokenizer for subtask obs with size {}".format(i))
                mlp_args = {
                    'input_size' : i, 
                    'units' : tokenizer_units + [num_features], 
                    'activation' : self.activation, 
                    'norm_func_name' : self.normalization,
                    'dense_func' : torch.nn.Linear,
                    'd2rl' : self.is_d2rl,
                    'norm_only_first_layer' : self.norm_only_first_layer
                }
                self.task_encoder_critic.append(self._build_mlp(**mlp_args))

            mlp_init = self.init_factory.create(**{"name": "default"})
            for nets in [self.self_encoder_critic, self.task_encoder_critic]:
                for m in nets.modules():
                    if isinstance(m, nn.Linear):
                        mlp_init(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)
            
            self.weight_token_critic = nn.Parameter(torch.zeros(1, 1, num_features)) # extra token
            self.weight_padding_critic = nn.Parameter(torch.zeros(1, 1, num_features)) # extra token

            d_model = num_features
            self.transformer_encoder_critic = AlternatingSelfCrossTransformer(
                d_model=d_model,
                nhead=params["transformer"]["layer_num_heads"],
                num_layers=params["transformer"]["num_layers"],       # e.g., 6 -> S, C, S, C, S, C
                dim_feedforward=params["transformer"]["layer_dim_feedforward"],
                dropout=drop_ratio,
                activation='relu',
                batch_first=True,
            )


            # weight init
            nn.init.trunc_normal_(self.weight_token, std=0.02)

            # extra mlp
            mlp_args = {
                'input_size': num_features,
                'units': params["transformer"]["extra_mlp_units"],
                'activation': self.activation,
                'dense_func': torch.nn.Linear,
            }

            output_size = 1
            comp_act = "identity"
            
            self.composer_critic = Composer(mlp_args, output_size=output_size, activation=comp_act)

            return
        
        def _eval_Transformer(self, obs, mask, mask_others):
            B = obs.shape[0]

            ###### self proprio
            self_obs = obs[..., :self.self_obs_size][mask] # [..., :223]
            self_token = self.self_encoder(self_obs).unsqueeze(1) # (B, 1, num_feats)
            ##############
            task_obs = obs[..., self.self_obs_size:][mask]
            
            ###### object
            obj_obs = task_obs[..., :self.obj_obs_size]
            obj_token = self.task_encoder[0](obj_obs).unsqueeze(1) # (B, 1, 64)

            ###### goal
            goal_obs = task_obs[..., self.obj_obs_size:self.obj_obs_size+self.goal_obs_size]
            goal_token = self.task_encoder[1](goal_obs).unsqueeze(1) * self.mult # (B, 1, 64)

            ##### other agents obs
            others_obs = task_obs[..., self.obj_obs_size+self.goal_obs_size:]
            others_obs = others_obs.view(-1, (self.num_agents), self.others_obs_each_size)

            weight_token = self.weight_token.expand(B, -1, -1)[mask]

            others_obs_min_self = others_obs[:,1:].clone() # other_obs[:, 0] is useless

            if mask_others is not None:
                ff_mask_others = ~mask_others[mask]
                others_obs_min_self[ff_mask_others] = 0

            # ---- zero pad if inference num_humanoids < num_humanoids during training ----
            # improved zero-zhot performance with this, compared to no_padding version.
            cur_n = others_obs.shape[1]
            target_n = self.system_max_agents

            if cur_n < target_n:
                pad_n = target_n - cur_n
                
                pad = torch.zeros(
                    others_obs_min_self.shape[0],  
                    pad_n,                         
                    others_obs_min_self.shape[2],  
                    dtype=others_obs_min_self.dtype,
                    device=others_obs_min_self.device
                )
                
                others_obs_min_self = torch.cat([others_obs_min_self, pad], dim=1)

            others_token = self.task_encoder[2](others_obs_min_self) # (B, Max_agents - 1, 64)
            
            # Feed forward
            x = torch.cat((weight_token, self_token, obj_token, goal_token), dim=1)
            x = self.transformer_encoder(x, kv=others_token)

            weights = self.composer(x[:, 0])

            return weights
        
        def _eval_Transformer_critic(self, obs, mask, mask_others):
            B = obs.shape[0]

            ###### self proprio
            self_obs = obs[..., :self.self_obs_size][mask] # [..., :223]
            self_token = self.self_encoder_critic(self_obs).unsqueeze(1) # (B, 1, num_feats)

            ##############
            task_obs = obs[..., self.self_obs_size:][mask]
            
            ###### object
            obj_obs = task_obs[..., :self.obj_obs_size]
            obj_token = self.task_encoder_critic[0](obj_obs).unsqueeze(1) # (B, 1, 64)

            ###### goal
            goal_obs = task_obs[..., self.obj_obs_size:self.obj_obs_size+self.goal_obs_size]
            goal_token = self.task_encoder_critic[1](goal_obs).unsqueeze(1) * self.mult # (B, 1, 64)

            ##### other agents obs
            others_obs = task_obs[..., self.obj_obs_size+self.goal_obs_size:]
            others_obs = others_obs.view(-1, (self.num_agents), self.others_obs_each_size)

            weight_token = self.weight_token_critic.expand(B, -1, -1)[mask]

            others_obs_min_self = others_obs[:,1:].clone()

            if mask_others is not None:
                ff_mask_others = ~mask_others[mask]
                others_obs_min_self[ff_mask_others] = 0

            # ---- zero pad if inference num_humanoids < num_humanoids during training ----
            # improved zero-zhot performance with this, compared to no_padding version.
            cur_n = others_obs.shape[1]
            target_n = self.system_max_agents

            if cur_n < target_n:
                pad_n = target_n - cur_n
                
                pad = torch.zeros(
                    others_obs_min_self.shape[0],  
                    pad_n,                         
                    others_obs_min_self.shape[2],  
                    dtype=others_obs_min_self.dtype,
                    device=others_obs_min_self.device
                )
                
                others_obs_min_self = torch.cat([others_obs_min_self, pad], dim=1)

            others_token = self.task_encoder_critic[2](others_obs_min_self) # (B, Max_agents - 1, 64)

            # feedforward
            x = torch.cat((weight_token, self_token, obj_token, goal_token), dim=1)
            x = self.transformer_encoder_critic(x, kv=others_token)
            weights = self.composer_critic(x[:, 0])

            return weights

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            mask, mask_others = obs_dict['mask'], obs_dict['mask_others']

            actor_outputs = self.eval_actor(obs, mask = mask, mask_others = mask_others)
            value = self.eval_critic(obs, mask = mask, mask_others = mask_others) # TODO: modify critic network

            output = actor_outputs + (value, states)

            return output
        
        def eval_actor(self, obs, mask, mask_others):

            if self.is_continuous and self.space_config['fixed_sigma']:
                
                mu = self._eval_Transformer(obs, mask, mask_others)
                sigma = mu * 0.0 + self.sigma_act(self.sigma)

                return mu, sigma

            else:
                raise NotImplementedError
    
        def eval_composer(self, obs, mask, mask_others):
            if self.type_id == 2:
                weights = self._eval_Transformer(obs, mask, mask_others)
            else:
                raise NotImplementedError

            return weights
    
        def eval_critic(self, obs, mask, mask_others):

            values = self._eval_Transformer_critic(obs, mask, mask_others)

            return values
        
        def eval_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits
        
        def eval_disc_masked(self, amp_obs):
            disc_mlp_out = self._disc_mlp_masked(amp_obs)
            disc_logits = self._disc_logits_masked(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights_masked(self):
            return torch.flatten(self._disc_logits_masked.weight)

        def get_disc_weights_masked(self):
            weights = []
            for m in self._disc_mlp_masked.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits_masked.weight))
            return weights
        
        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights
        
        def _build_disc(self, input_shape):
            self._disc_mlp = nn.Sequential()

            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._disc_mlp = self._build_mlp(**mlp_args)
            
            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias) 


            # Add masked
            self._disc_mlp_masked = nn.Sequential()

            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._disc_mlp_masked = self._build_mlp(**mlp_args)
            
            mlp_out_size = self._disc_units[-1]
            self._disc_logits_masked = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp_masked.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits_masked.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits_masked.bias) 


            return
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, kv, *, attn_mask=None, kv_key_padding_mask=None):
        # Cross-attn: Q from q, K/V from kv
        y, _ = self.mha(q, kv, kv, attn_mask=attn_mask, key_padding_mask=kv_key_padding_mask, need_weights=False)
        x = self.norm1(q + self.dropout1(y))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x

class AlternatingSelfCrossTransformer(nn.Module):

    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=True):
        super().__init__()

        # alternate: 0 -> self, 1 -> cross, 2 -> self, ...
        layers = []
        for i in range(num_layers):
            if i % 2 == 0:
                layers.append(nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=dim_feedforward, dropout=dropout,
                    activation=activation, batch_first=batch_first
                ))
            else:
                layers.append(CrossAttentionBlock(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=dim_feedforward, dropout=dropout,
                    activation=activation, batch_first=batch_first
                ))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x_q,                         # (B, Tq, d)
        kv=None,                     # (B, Tk, d) context for cross-attn layers
        *,
        q_key_padding_mask=None,     # (B, Tq) True = pad in queries
        kv_key_padding_mask=None,    # (B, Tk) True = pad in kv (for cross)
        cross_attn_mask=None,        # (Tq, Tk) or (B*nhead, Tq, Tk) for cross layers
        self_attn_mask=None          # (Tq, Tq) or (B*nhead, Tq, Tq) for self layers
    ):
        x = x_q
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.TransformerEncoderLayer):
                # self-attention on queries
                x = layer(x, src_mask=self_attn_mask, src_key_padding_mask=q_key_padding_mask)
            else:
                # cross-attention: queries attend to kv
                if kv is None:
                    raise ValueError("kv must be provided for cross-attention layers.")
                x = layer(x, kv, attn_mask=cross_attn_mask, kv_key_padding_mask=kv_key_padding_mask)
        return x