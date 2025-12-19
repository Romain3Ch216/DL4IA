import torch
import torch.nn as nn
import torch.nn.functional as F


class Automap(nn.Module):
    def __init__(self, m, K, dim_bottleneck=64, conv_channels=64):
        """PyTorch implementation of AUTOMAP
        Zhu, B., Liu, J. Z., Cauley, S. F., Rosen, B. R., & Rosen, M. S. (2018). 
        Image reconstruction by domain-transform manifold learning. Nature, 555(7697), 487-492
        """
        super().__init__()

    def forward(self, kspace, mask):
        raise NotImplementedError