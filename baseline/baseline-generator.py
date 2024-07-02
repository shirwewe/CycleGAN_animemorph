import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}\n")

#Generator
#Running into an error w/ numpy version:
"""
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.0 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
"""
#Downgrading does NOT work

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*32, 4, 1, 0), # N x f_g*32 x 4 x 4
            self._block(features_g*32, features_g*16, 4, 2, 1), #8 x 8
            self._block(features_g*16, features_g*8, 4, 2, 1), #16 x 16
            self._block(features_g*8, features_g*4, 4, 2, 1), #32 x 32
            self._block(features_g*4, features_g*2, 4, 2, 1), #64 x 64
            self._block(features_g*2, features_g, 4, 2, 1), #128 x 128
            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1,), # 256 x 256
            nn.Tanh(), #[-1, 1]
        )

        #TO DO: Change so output is 256x256
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):

        for i, layer in enumerate(self.gen):
            x = layer(x)
            print(f"Layer {i} output size: {x.size()}")
        print("\n")
        return x

#initialize weight for particular model
def initialize_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

#to be deleted later
def test():
    N, in_channels, H, W = 8,3,256,256
    z_dim = 100
    gen = Generator(z_dim, in_channels, 32)
    initialize_weight(gen)
    z = torch.rand((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("No errors?")

test()