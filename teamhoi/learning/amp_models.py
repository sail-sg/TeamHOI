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

import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

class ModelAMPContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('amp', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelAMPContinuous.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            #TODO: 
            result = super().forward(input_dict)

            if (is_train):
                amp_obs = input_dict['amp_obs']
                if amp_obs is not None:
                    disc_agent_logit = self.a2c_network.eval_disc(amp_obs)
                    result["disc_agent_logit"] = disc_agent_logit
                else:
                    result["disc_agent_logit"] = None

                amp_obs_replay = input_dict['amp_obs_replay']
                if amp_obs_replay is not None:
                    disc_agent_replay_logit = self.a2c_network.eval_disc(amp_obs_replay)
                    result["disc_agent_replay_logit"] = disc_agent_replay_logit
                else:
                    result["disc_agent_replay_logit"] = None

                amp_demo_obs = input_dict['amp_obs_demo']
                if amp_demo_obs is not None:
                    disc_demo_logit = self.a2c_network.eval_disc(amp_demo_obs)
                    result["disc_demo_logit"] = disc_demo_logit
                else:
                    result["disc_demo_logit"] = None


                ### Add masked
                amp_obs_masked = input_dict['amp_obs_masked']
                disc_agent_logit_masked = self.a2c_network.eval_disc_masked(amp_obs_masked)
                result["disc_agent_logit_masked"] = disc_agent_logit_masked

                amp_obs_replay_masked = input_dict['amp_obs_replay_masked']
                disc_agent_replay_logit_masked = self.a2c_network.eval_disc_masked(amp_obs_replay_masked)
                result["disc_agent_replay_logit_masked"] = disc_agent_replay_logit_masked

                amp_demo_obs_masked = input_dict['amp_obs_demo_masked2']
                disc_demo_logit_masked = self.a2c_network.eval_disc_masked(amp_demo_obs_masked)
                result["disc_demo_logit_masked"] = disc_demo_logit_masked


            return result