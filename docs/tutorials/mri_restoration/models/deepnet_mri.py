import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepNetMRI(nn.Module):
    """
    Re-implementation of DeepNetMRI for Dynamic MRI Reconstruction:
    https://github.com/js3611/Deep-MRI-Reconstruction/tree/master
    https://arxiv.org/pdf/1703.00555

    Quote from the paper:
    "Our CNN takes in a two-channeled image $\mathbb{R}^{\sqrt{n} \times \sqrt{n} \times 2},
    where each channel stores real and imaginary parts of the undersampled image.
    Based on literature, we used the following network architecture for CNN, illus-
    trated in Figure 1: it has $n_d - 1$ convolution layers $C_i$, which are all followed by
    Rectifier Linear Units (ReLU) as a choice of nonlinearity. For each of them, we
    used a kernel size k = 3 [23] and the number of filters were set to nf = 64. The
    network is followed by another convolution layer $C_{rec}$ with kernel size k = 3 and
    $n_f$ = 2, which projects the extracted representation back to image domain. We
    also used residual connection [5], which sums the output of the CNN module
    with its input. Finally, we form a cascading network by using the DC layers
    interleaved with the CNN reconstruction modules. For our experiment, we chose
    $n_d$ = 5 and $n_c$ = 5." $n_c$ is the number of cascaded CNNs.
    """
    def __init__(self, n_channels=2, nc=5, nd=5, nf=64, kernel_size=3, noise_level=None):
        super(DeepNetMRI, self).__init__()

        self.n_channels = n_channels

        self.cascaded_cnns = nn.ModuleList([
            CNN(n_channels, nd, nf, kernel_size) for _ in range(nc)
        ])

        self.dc_layer = DataConsistency(noise_level)

    def forward(self, x, kspace, mask):
        raise NotImplementedError


class DataConsistency(nn.Module):
    """ Data consistency layer as described in https://arxiv.org/pdf/1703.00555.
    Re-implementation of DataConsistencyInKspace in 
    https://github.com/js3611/Deep-MRI-Reconstruction/blob/master/cascadenet_pytorch/kspace_pytorch.py.
    """
    def __init__(self, noise_level):
        super(DataConsistency, self).__init__()
        self.noise_level = noise_level

    def forward(self, image: torch.Tensor, kspace: torch.Tensor, mask: torch.Tensor):
        kspace = kspace.unsqueeze(0)
        mask = mask.unsqueeze(0).long()

        image = ...
        k = ...
        out = ...

        image_res = torch.fft.ifftn(out)
        image_res = torch.view_as_real(image_res).float()
        image_res = image_res.permute(0, 3, 1, 2)
        return image_res
      

class CNN(nn.Module):
    """2D convolutional network with one residual connection.
    """
    def __init__(self, n_channels, nd=5, nf=64, kernel_size=3):
        super(CNN, self).__init__()
        assert nd > 1, "CNN should have at least 2 layers."
        
        in_layer = nn.Sequential(
            ...
        )

        out_layer = nn.Conv2d(
            ...
        )

        if nd == 2:
            self.layers = nn.ModuleList([in_layer, out_layer])
        else:
            int_layers = [
                nn.Sequential(
                    ...
                ) for _ in range(nd - 2)
            ]
            self.layers = nn.ModuleList([in_layer] + int_layers + [out_layer])

    def forward(self, x):
        raise NotImplementedError