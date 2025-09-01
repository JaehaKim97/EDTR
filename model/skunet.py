import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms


class Selector(nn.Module):
    def __init__(self, channel, reduction=16, crp_classify=False):
        super(Selector, self).__init__()
        self.spatial_attention = 4
        self.in_channel = channel * (self.spatial_attention ** 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((self.spatial_attention, self.spatial_attention))

        self.fc = nn.Sequential(
            nn.Linear(self.in_channel, self.in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.att_conv1 = nn.Linear(self.in_channel // reduction, self.in_channel)
        self.att_conv2 = nn.Linear(self.in_channel // reduction, self.in_channel)

    def forward(self, x):

        b, c, H, W = x.size()

        y = self.avg_pool(x).view(b, -1)
        y = self.fc(y)

        att1 = self.att_conv1(y).view(b, c, self.spatial_attention, self.spatial_attention)
        att2 = self.att_conv2(y).view(b, c, self.spatial_attention, self.spatial_attention)

        attention = torch.stack((att1, att2))
        attention = nn.Softmax(dim=0)(attention)

        att1 = F.interpolate(attention[0], scale_factor=(H / self.spatial_attention, W / self.spatial_attention), mode="nearest")
        att2 = F.interpolate(attention[1], scale_factor=(H / self.spatial_attention, W / self.spatial_attention), mode="nearest")

        return att1, att2


class SelectiveConv(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels, first=False):
        super(SelectiveConv, self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.selector = Selector(out_channels, reduction=reduction)
        self.IN = nn.InstanceNorm2d(in_channels)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        if self.first:
            f_input = x
            s_input = x
        else:
            f_input = self.BN(x)
            f_input = self.relu(f_input)

            s_input = self.IN(x)
            s_input = self.relu(s_input)

        out1 = self.conv1(f_input)
        out2 = self.conv2(s_input)

        out = out1 + out2

        att1, att2 = self.selector(out)
        out = torch.mul(out1, att1) + torch.mul(out2, att2)

        return out


class SKDown(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels, first=False):
        super(SKDown, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SelectiveConv(kernel_size, padding, bias, reduction, in_channels, out_channels, first=first)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SKUp(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = SelectiveConv(kernel_size, padding, bias, reduction, in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        pass

    def forward(self, x):
        pass
    

class Normalize:
    def __init__(self):
        self.transforms = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.inv_transforms = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def __call__(self, img, inv=False):
        if inv:
            return self.inv_transforms(img)
        else:
            return self.transforms(img)


class SKUNet(nn.Module):
    def __init__(self, bilinear=True):
        super(SKUNet, self).__init__()
        self.bilinear = bilinear

        self.down1 = nn.Conv2d(kernel_size=9, padding=4,in_channels=3, out_channels=32)
        self.down2 = SKDown(3, 1, False, 16, 32, 64)
        self.down3 = SKDown(3, 1, False, 16, 64, 64)
        self.up1 = SKUp(3, 1, False, 16, 128, 32, bilinear)
        self.up2 = SKUp(3, 1, False, 16, 64, 16, bilinear)
        self.up3 = nn.Conv2d(kernel_size=3, padding=1, in_channels=16, out_channels=3)
        self.normalize = Normalize()

    def forward(self, x, normalize=True):
        if normalize:
            x = self.normalize(x)
        
        x_origin = x
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x)

        return torch.add(x, x_origin) #, x
