import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.ppo.policies import CnnPolicy

def orthogonal_init(layer, gain=nn.init.calculate_gain("relu")):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

class CustomCnnPolicy(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCnnPolicy, self).__init__(*args, **kwargs)

        # Apply orthogonal init to the entire network
        self.apply(orthogonal_init)