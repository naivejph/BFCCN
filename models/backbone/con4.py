import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设从 mods.py 导入了 HFRM
from models.backbone.mods import HFRM

class ConvBlock(nn.Module):
    
    def __init__(self, input_channel, output_channel):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel))

    def forward(self, inp):
        return self.layers(inp)


class BackBone(nn.Module):

    def __init__(self, num_channel=64):
        super().__init__()
        
        self.layers = nn.Sequential(
            ConvBlock(3, num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel, num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel, num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel, num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # 添加 HFRM 模块
        self.hfrm = HFRM(num_channel, num_channel)  # 确保 HFRM 的输入和输出通道数与 BackBone 匹配

    def forward(self, inp):
        # 传递通过 BackBone 的特征到 HFRM
        features = self.layers(inp)
        output = self.hfrm(features)
        return output
