import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = x
        return self.block(x) + residual


class ResizeBlock(nn.Module):
    def __init__(self, size, in_channels=3, residual_num=1):
        super(ResizeBlock, self).__init__()

        self.size = size

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16)
        )

        residual_layers = []
        for i in range(residual_num):
            residual_layers.append(ResBlock(16, 16))
        self.residual_blocks = nn.Sequential(*residual_layers)

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )

        self.block4 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x1 = F.interpolate(x, size=self.size, mode="bilinear",
                           align_corners=False, recompute_scale_factor=False)

        x2 = self.block1(x)
        x3 = self.block2(x2)
        x4 = F.interpolate(x3, size=self.size, mode="bilinear",
                           align_corners=False, recompute_scale_factor=False)
        x5 = self.residual_blocks(x4)
        x6 = self.block3(x5)
        x7 = self.block4(x6+x4)
        return x7 + x1