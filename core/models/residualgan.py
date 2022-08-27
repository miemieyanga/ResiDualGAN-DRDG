from turtle import shape
import torch.nn as nn
import torch
from .resize_block import ResizeBlock
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class Generator(nn.Module):
    def __init__(self, channels=3, k=1.0, k_grad=False):
        super(Generator, self).__init__()

        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(nn.ConvTranspose2d(128, channels, 4, stride=2, padding=1),nn.Tanh())
        self.k = torch.nn.Parameter(torch.tensor([k]), requires_grad=k_grad)
        
        self.dup1 = UNetUp(512, 512, dropout=0.5)
        self.dup2 = UNetUp(1024, 512, dropout=0.5)
        self.dup3 = UNetUp(1024, 512, dropout=0.5)
        self.dup4 = UNetUp(1024, 256)
        self.dup5 = UNetUp(512, 128)
        self.dup6 = UNetUp(256, 64)
        self.dfinal = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1), nn.Sigmoid())

        
    def forward(self, x, require_depth=False):
        # Propogate noise through fc layer and reshape to img shape
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        
        if require_depth:
            du1 = self.up1(d7, d6)
            du2 = self.up2(du1, d5)
            du3 = self.up3(du2, d4)
            du4 = self.up4(du3, d3)
            du5 = self.up5(du4, d2)
            du6 = self.up6(du5, d1)
            return self.final(u6) + self.k * x, self.dfinal(du6)
        else:
            return self.final(u6) + self.k * x

    
class LinkNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(LinkNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x
    
class ResizeGenerator(nn.Module):
    def __init__(self, in_channels, size, generator="UNet", resize_block=False, 
    interpolation="bilinear", k=1.0, residual_blocks=5, k_grad=False, post_resize=True):
        super(ResizeGenerator, self).__init__()
        self.channels = in_channels
        self.size = size
        self.interpolation = interpolation
        self.post_resize = post_resize
        
        if generator=="UNet":
            self.generator = Generator(in_channels, k, k_grad=k_grad)
        # TODO: k
        elif generator=="LinkNet":
            self.generator = GeneratorLink(in_channels)
        elif generator=="ResNet":
            self.generator = GeneratorResNet(in_channels, residual_blocks)
        else:
            raise NotImplementedError
        
        if resize_block:
            self.resize_block = ResizeBlock(size, in_channels)
        else:
            self.resize_block = None

    def _resize(self,x):
        if self.resize_block:
            x = self.resize_block(x)
        else:
            x = F.interpolate(x, size=self.size, mode=self.interpolation, recompute_scale_factor=False)

        return x
        
    def forward(self, x, require_depth=False):
        if not self.post_resize:
            x = self._resize(x)

        if not require_depth:
            x = self.generator.forward(x, require_depth)
        else:
            x, depth = self.generator.forward(x, require_depth)

        if self.post_resize:
            x = self._resize(x)

        if require_depth:
            return x, depth

        return x
        
        
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discrimintor_block(in_features, out_features, normalize=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discrimintor_block(in_channels, 64, normalize=False),
            *discrimintor_block(64, 128),
            *discrimintor_block(128, 256),
            *discrimintor_block(256, 512),
            *discrimintor_block(512, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4)
        )

    def forward(self, img):
        return self.model(img)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, channels, num_residual_blocks, k=1.0, k_grad=False):
        super(GeneratorResNet, self).__init__()
        self.k = torch.nn.Parameter(torch.tensor([k]), requires_grad=k_grad)

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 3),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x, require_depth=False):
        if require_depth:
            raise NotImplementedError("resnet for requiring depth")
        return self.model(x) + x


class GeneratorLink(nn.Module):
    def __init__(self, channels = 3, k=1.0, k_grad=False):
        super(GeneratorLink, self).__init__()
        self.k = torch.nn.Parameter(torch.tensor([k]), requires_grad=k_grad)
        
        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = LinkNetUp(512, 512, dropout=0.5)
        self.up2 = LinkNetUp(512, 512, dropout=0.5)
        self.up3 = LinkNetUp(512, 512, dropout=0.5)
        self.up4 = LinkNetUp(512, 256)
        self.up5 = LinkNetUp(256, 128)
        self.up6 = LinkNetUp(128, 64)

        self.final = nn.Sequential(nn.ConvTranspose2d(64, channels, 4, stride=2, padding=1), nn.Tanh())

    def forward(self, x, require_depth=False):
        if require_depth:
            raise NotImplementedError("linknet for requiring depth")
            
        d1 = self.down1(x)     # 64
        d2 = self.down2(d1)    # 128
        d3 = self.down3(d2)    # 256
        d4 = self.down4(d3)    # 512
        d5 = self.down5(d4)    # 512
        d6 = self.down6(d5)    # 512
        d7 = self.down7(d6)    # 512
        u1 = self.up1(d7) + d6    # 512->512
        u2 = self.up2(u1) + d5    # 512->512
        u3 = self.up3(u2) + d4    # 512->512
        u4 = self.up4(u3) + d3    # 512->256
        u5 = self.up5(u4) + d2    # 256->128
        u6 = self.up6(u4) + d1    # 128->64

        return self.final(u6) + self.k * x
