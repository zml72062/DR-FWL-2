import torch
import torch.nn as nn

class Normalization(nn.Module):
    def __init__(self,
                 out_channels: int,
                 norm_type: str):
        super().__init__()
        self.out_channels = out_channels
        self.norm_type = norm_type

        if self.norm_type == "none":
            self.norm = torch.nn.Identity()
        elif self.norm_type == 'batch_norm':
            self.norm = torch.nn.BatchNorm1d(out_channels)
        elif self.norm_type == 'layer_norm':
            self.norm = torch.nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.norm, 'reset_parameters'):
            self.norm.reset_parameters()

    def forward(self, x):
        return self.norm(x)