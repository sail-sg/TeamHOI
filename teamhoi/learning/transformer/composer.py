from rl_games.algos_torch.network_builder import NetworkBuilder
import torch
import torch.nn as nn

class Composer(NetworkBuilder.BaseNetwork):
    def __init__(self, mlp_args, output_size, activation="sigmoid", last_layer_all_zero_init=False):
        super(Composer, self).__init__()

        self.actors = self._build_sequential_mlp(output_size, **mlp_args)
        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "identity":
            self.act = nn.Identity()
        else:
            raise NotImplementedError

        mlp_init = self.init_factory.create(**{"name": "default"})
        for m in self.actors.modules():
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        
        if last_layer_all_zero_init:
            torch.nn.init.zeros_(self.actors[-1].weight)
            torch.nn.init.zeros_(self.actors[-1].bias)

        return

    def _build_sequential_mlp(self, actions_num, input_size, units, activation, dense_func):
        print('build mlp:', input_size)
        in_size = input_size
        layers = []
        for unit in units:
            layers.append(dense_func(in_size, unit))
            layers.append(self.activations_factory.create(activation))
            
            in_size = unit

        layers.append(nn.Linear(units[-1], actions_num))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.act(self.actors(x))
