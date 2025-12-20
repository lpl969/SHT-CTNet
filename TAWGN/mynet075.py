#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)

        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)

        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)

        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape

        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)

        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=16, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        self.n_iter = n_iter
        self.stoken_size = stoken_size

        self.scale = dim ** - 0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)

        self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=proj_drop)

    def stoken_forward(self, x):
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size

        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x.shape

        hh, ww = H // h, W // w

        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww))

        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)

        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale

                affinity_matrix = affinity_matrix.softmax(-1)

                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)

                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix

                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(
                        B, C, hh, ww)

                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix

        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)

        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)

        stoken_features = self.stoken_refine(stoken_features)

        stoken_features = self.unfold(stoken_features)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)

        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)

        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]

        return pixel_features

    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        stoken_features = self.stoken_refine(stoken_features)
        return stoken_features

    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)

class Pconv(nn.Module):
    def __init__(self, dim, n_div, forward='split_cat', kernel_size=7):
        super().__init__()
        self.dim = dim
        self.n_div = n_div
        self.forward_method = forward
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList(
            [nn.Conv2d(dim // n_div, dim // n_div, kernel_size=kernel_size, padding=kernel_size // 2) for _ in range(n_div)]
        )

    def forward(self, x):
        if self.forward_method == 'split_cat':
            split_x = torch.chunk(x, self.n_div, dim=1)
            out = [conv(split_x[i]) for i, conv in enumerate(self.convs)]
            return torch.cat(out, dim=1)
        else:
            raise ValueError(f"Unknown forward method: {self.forward_method}")

class SE_Module(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE_Module, self).__init__()
        reduced_channels = max(channel // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channel, reduced_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channel, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x

class LFAEncoder(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4.0, kernel_s=7, downsample=True):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=3//2, groups=dim)

        self.dwconv2 = Pconv(dim=dim, n_div=4, forward='split_cat', kernel_size=kernel_s)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(expan_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(expan_ratio * dim), dim)

        self.se = SE_Module(channel=dim, ratio=16)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.downsample = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv2(self.dwconv1(x) + input)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = x.permute(0, 3, 1, 2)
        x = self.se(input + self.drop_path(x))
        x = self.downsample(x)
        return x
class LFAStokenNetwork(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4.0, kernel_s=7,downsample=True):
        super().__init__()
        self.encoder = LFAEncoder(dim, drop_path, layer_scale_init_value, expan_ratio, kernel_s,downsample=downsample)
        self.stoken_attention = StokenAttention(dim, stoken_size, n_iter, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

    def forward(self, x):
        shortcut = x
        x = self.encoder(x) + shortcut
        shortcut = x
        x = self.stoken_attention(x) + shortcut

        return x
class mynet(nn.Module):

    def __init__(self,  num_classes):
        self.inplanes = 32
        super(mynet, self).__init__()


        self.conv = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=(3, 3))
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=(1,1))
        self.BN = nn.BatchNorm2d(16)
        
        self.relu = nn.ReLU(inplace=True)
        self.LS = LFAStokenNetwork(16, stoken_size=[4,4],downsample=False)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=(1,1))
        self.BN1 = nn.BatchNorm2d(32)
        self.LS1 = LFAStokenNetwork(32, stoken_size=[8,8],downsample=False)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.LazyLinear(num_classes)


    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.LS(x)
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.LS1(x)





        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
model = mynet(7)
from torchsummary import summary
if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(summary(model, (1, 224, 224)))