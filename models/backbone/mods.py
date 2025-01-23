import torch
from torch import nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
       return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size//2), bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, kernel_size=5):
        super(RDB, self).__init__()
        self.conv1 = default_conv(64, 64, kernel_size)
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate, kernel_size=kernel_size) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1) #k=1

    def forward(self, x, lrl=True):
        if lrl:
            return x + self.lff(self.layers(x)) # local residual learning
        else:
            return self.layers(x)
 
class HFRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFRM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 初始化卷积层和RDB模块
        self.conv1 = default_conv(in_channels, out_channels, kernel_size=5)
        self.RDB_1 = nn.Sequential(RDB(out_channels, out_channels, 3), nn.ReLU(True))
        self.RDB_2 = nn.Sequential(RDB(out_channels, out_channels, 3), nn.ReLU(True))
        self.conv_block2 = nn.Sequential(default_conv(out_channels, out_channels, 5), nn.ReLU(True))
        # 将输出调整到与输入相同的通道数
        self.final_conv = default_conv(out_channels, in_channels, kernel_size=5)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.RDB_1(out1)
        out3 = self.RDB_2(out2)
        out4 = out1 + out2 + out3
        out5 = self.conv_block2(out4)
        # 最后一步调整通道数，保证与输入维度相同
        out = self.final_conv(out5)
        # 执行相减操作
        return x - out

