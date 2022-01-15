import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class UDVD(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(19, 128, 3, 1, 1)
        body = [ResBlock(128, 3, 0.1) for _ in range(15)]
        self.body = nn.Sequential(*body)
        self.UpDyConv = UpDynamicConv()
        self.ComDyConv1 = CommonDynamicConv()
        self.ComDyConv2 = CommonDynamicConv()

    def forward(self, image, kernel, noise):
        assert image.size(1) == 3, 'Channels of Image should be 3, not {}'.format(image.size(1))
        assert kernel.size(1) == 15, 'Channels of kernel should be 15, not {}'.format(kernel.size(1))
        assert noise.size(1) == 1, 'Channels of noise should be 1, not {}'.format(noise.size(1))
        inputs = torch.cat([image, kernel, noise], 1)
        head = self.head(inputs)
        body = self.body(head) + head
        output1 = self.UpDyConv(image, body)
        output2 = self.ComDyConv1(output1, body)
        output3 = self.ComDyConv2(output2, body)
        return output1, output2, output3

    def cal_params(self):
        params = list(self.parameters())
        k = 0
        for i in params:
            l = 1
            for j in i.size():
                l *= j
            k = k + l
        print("Total parameters is :" + str(k))

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, res_scale=1.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, 1, padding)
        )
        self.res_scale = res_scale

    def forward(self, inputs):
        return inputs + self.conv(inputs) * self.res_scale


class PixelConv(nn.Module):
    def __init__(self, scale=2, depthwise=False):
        super().__init__()
        self.scale = scale
        self.depthwise = depthwise

    def forward(self, feature, kernel):
        NF, CF, HF, WF = feature.size()
        NK, ksize, HK, WK = kernel.size()
        assert NF == NK and HF == HK and WF == WK
        if self.depthwise:
            ink = CF
            outk = 1
            ksize = int(np.sqrt(int(ksize // (self.scale ** 2))))
            pad = (ksize - 1) // 2
        else:
            ink = 1
            outk = CF
            ksize = int(np.sqrt(int(ksize // CF // (self.scale ** 2))))
            pad = (ksize - 1) // 2

        # features unfold and reshape, same as PixelConv
        feat = F.pad(feature, [pad, pad, pad, pad])
        feat = feat.unfold(2, ksize, 1).unfold(3, ksize, 1)
        feat = feat.permute(0, 2, 3, 1, 5, 4).contiguous()
        feat = feat.reshape(NF, HF, WF, ink, -1)

        # kernel
        kernel = kernel.permute(0, 2, 3, 1).reshape(NK, HK, WK, ksize * ksize, self.scale ** 2 * outk)

        output = torch.matmul(feat, kernel)
        output = output.permute(0, 3, 4, 1, 2).view(NK, -1, HF, WF)
        if self.scale > 1:
            output = F.pixel_shuffle(output, self.scale)
        return output


class CommonDynamicConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1)
        )
        # I'm not sure how to deal the feautre.
        # Because it need to upsample the feature and align,
        # but the paper not provide useful information about it, just provide
        # Sub-pixel Convolution layer is used to align the resolutions between paths.
        self.feat_conv = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(32, 128, 1)
        )
        self.feat_residual = nn.Sequential(
            nn.Conv2d(160, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1)
        )
        self.feat_kernel = nn.Conv2d(160, 25, 3, 1, 1)
        self.pixel_conv = PixelConv(scale=1, depthwise=True)

    def forward(self, image, features):
        image_conv = self.image_conv(image)
        features = self.feat_conv(features)
        cat_inputs = torch.cat([image_conv, features], 1)

        kernel = self.feat_kernel(cat_inputs)
        output = self.pixel_conv(image, kernel)

        residual = self.feat_residual(cat_inputs)
        return output + residual


class UpDynamicConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1)
        )
        self.feat_residual = nn.Sequential(
            nn.Conv2d(160, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(16, 3, 3, 1, 1)
        )
        self.feat_kernel = nn.Conv2d(160, 25 * 4, 3, 1, 1)
        self.pixel_conv = PixelConv(scale=2, depthwise=True)

    def forward(self, image, features):
        image_conv = self.image_conv(image)
        cat_inputs = torch.cat([image_conv, features], 1)

        kernel = self.feat_kernel(cat_inputs)
        output = self.pixel_conv(image, kernel)

        residual = self.feat_residual(cat_inputs)
        return output + residual


def demo():
    net = UDVD()
    net.cal_params()
    inputs = torch.randn(1, 3, 64, 64)
    kernel = torch.randn(1, 15, 64, 64)
    noise = torch.randn(1, 1, 64, 64)

    with torch.no_grad():
        output1, output2, output3 = net(inputs, kernel, noise)

    print(output1.size())
    print(output2.size())
    print(output3.size())


if __name__ == '__main__':
    demo()