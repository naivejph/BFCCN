import torch
import sys
sys.path.append('../../../../')
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import Conv_4, ResNet
import math
from utils.l2_norm import l2_norm
from BfCCN.models.modules.DLE import SelfCorrelationComputation, DLE
from models.modules.BFM import BFM
from BfCCN.models.modules.process import process

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Our(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.shots = [self.args.train_shot, self.args.train_query_shot]
        self.way = self.args.train_way
        self.resnet = self.args.resnet
        self.disturb_num = self.args.disturb_num
        self.short_cut_weight = self.args.short_cut_weight
        self.resolution = 25

        if self.resnet:
            self.num_channel = 640
            self.num_channel2 = 160
            self.feature_extractor_1 = ResNet.resnet12(drop=True)
            self.feature_extractor_2 = ResNet.resnet12(drop=True)
            self.feature_size = 640
            self.pro = process(channels=self.num_channel) 
            self.conv_block3 = nn.Sequential(
                BasicConv(self.num_channel // 2, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True)
            )
            self.conv_block4 = nn.Sequential(
                BasicConv(self.num_channel, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True)
            )
            self.max3 = nn.AdaptiveMaxPool2d((1, 1))
            self.max4 = nn.AdaptiveMaxPool2d((1, 1))
            self.both_mlp2 = nn.Sequential(
                nn.BatchNorm1d(self.num_channel2 * self.disturb_num),
                nn.Linear(self.num_channel2 * self.disturb_num, self.num_channel2 * self.disturb_num),
                nn.ELU(inplace=True)
            )
            self.both_mlp4 = nn.Sequential(
                nn.BatchNorm1d(self.feature_size),
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
            self.mask_branch = nn.Sequential(
                nn.Conv2d(self.num_channel, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, self.disturb_num, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.num_channel = 64
            self.num_channel2 = 64

            self.feature_extractor_1 = Conv_4.BackBone(self.num_channel)
            self.feature_extractor_2 = Conv_4.BackBone(self.num_channel)
            self.pro = process(channels=self.num_channel) 
            self.feature_size = 64 *5 *5
            self.avg = nn.AdaptiveAvgPool2d((5, 5))
            self.both_mlp2 = nn.Sequential(
                nn.Linear(self.num_channel2 * self.disturb_num, self.num_channel2 * self.disturb_num),
                nn.ELU(inplace=True)
            )
            self.both_mlp3 = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
            self.both_mlp4 = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
            self.mask_branch = nn.Sequential(
                nn.Conv2d(self.num_channel, 30, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.Conv2d(30, self.disturb_num, kernel_size=1, stride=1, padding=0)
            )   

        self.bfm = BFM(channel=self.num_channel, kernel_size=7)
        self.dle_module = self._make_dle_layer(planes=[640, 64, 64, 64, 640])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)


    def _make_dle_layer(self, planes):
        stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        layers = list()
        corr_block = SelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
        self_block = DLE(planes=planes, stride=stride)
        layers.append(corr_block)
        layers.append(self_block)
        return nn.Sequential(*layers)
    

    def get_feature_map(self, inp):

        fa_3, fa_4 = self.feature_extractor_1(inp)
        fb_3, fb_4 = self.feature_extractor_2(inp)
        fa_4 = self.bfm(fa_4)
        fb_4 = self.bfm(fb_4)
        fa_3 = self.pro(fa_3)
        fb_3 = self.pro(fb_3)
        return fa_3, fa_4, fb_3, fb_4
    
    def integration(self, layer1, layer2):

        batch_size = layer1.size(0)
        channel_num = layer1.size(1)
        disturb_num = layer2.size(1)
        layer1 = layer1.unsqueeze(2)
        layer2 = layer2.unsqueeze(1)

        sum_of_weight = layer2.view(batch_size, disturb_num, -1).sum(-1) + 0.00001
        vec = (layer1 * layer2).view(batch_size, channel_num, disturb_num, -1).sum(-1)

        vec = vec / sum_of_weight.unsqueeze(1)
        vec = vec.view(batch_size, channel_num*disturb_num)
        return vec
    
    def get_cosine_dist(self, inp, way, shot):
        fa_3, fa_4, fb_3, fb_4= self.get_feature_map(inp)

        fb_3 = self.dle_module(fb_3)
        fb_3 = F.normalize(fb_3, dim=1, p=2)
        fb_4 = self.dle_module(fb_4)
        fb_4 = F.normalize(fb_4, dim=1, p=2)

        heat_map_a = nn.functional.interpolate(fa_4, size=(fa_3.shape[-1], fa_3.shape[-1]), mode='bilinear', align_corners=False)
        heat_map_a = self.mask_branch(heat_map_a)
        mask_a = torch.sigmoid(heat_map_a)
        fa_vec0 = self.integration(fa_3, mask_a)
        fa_vec = (1 - self.short_cut_weight) *self.both_mlp2(fa_vec0) + self.short_cut_weight * fa_vec0
        support_a2 = fa_vec[:way * shot].view(way, shot, -1).mean(1)
        query_a2 = fa_vec[way * shot:]
        cos_a1 = F.linear(l2_norm(query_a2), l2_norm(support_a2))
        
        
        if self.resnet:
            fa_4 = self.conv_block4(fa_4)
            fa_4 = self.max4(fa_4)
            fa_4 = fa_4.view(fa_4.size(0), -1)
        else:
            fa_4 = fa_4.view(fa_4.size(0), -1)


        fa_4 = (1 - self.short_cut_weight) * self.both_mlp4(fa_4) + self.short_cut_weight * fa_4


        support_a4 = fa_4[:way * shot].view(way, shot, -1).mean(1)
        query_a4 = fa_4[way * shot:]
        cos_a2 = F.linear(l2_norm(query_a4), l2_norm(support_a4))

#################################################################################################
        heat_map_b = nn.functional.interpolate(fb_4, size=(fb_3.shape[-1], fb_3.shape[-1]), mode='bilinear', align_corners=False)
        heat_map_b = self.mask_branch(heat_map_b)
        mask_b = torch.sigmoid(heat_map_b)
        fb_vec0 = self.integration(fb_3, mask_b)
        fb_vec = (1 - self.short_cut_weight) *self.both_mlp2(fb_vec0) + self.short_cut_weight * fb_vec0
        support_b2 = fb_vec[:way * shot].view(way, shot, -1).mean(1)
        query_b2 = fb_vec[way * shot:]
        cos_b1 = F.linear(l2_norm(query_b2), l2_norm(support_b2))
        
        
        if self.resnet:
            fb_4 = self.conv_block4(fb_4)
            fb_4 = self.max4(fb_4)
            fb_4 = fb_4.view(fb_4.size(0), -1)
        else:
            fb_4 = fb_4.view(fb_4.size(0), -1)


        fb_4 = (1 - self.short_cut_weight) * self.both_mlp4(fb_4) + self.short_cut_weight * fb_4


        support_b4 = fb_4[:way * shot].view(way, shot, -1).mean(1)
        query_b4 = fb_4[way * shot:]
        cos_b2 = F.linear(l2_norm(query_b4), l2_norm(support_b4))

        return cos_a1, cos_a2, cos_b1, cos_b2

    def meta_test(self, inp, way, shot):
        cos_a1, cos_a2, cos_b1, cos_b2= self.get_cosine_dist(inp=inp, way=way, shot=shot)
        scores = cos_a1 + cos_a2 + cos_b1 + cos_b2


        _, max_index = torch.max(scores, 1)
        return max_index

    def forward(self, inp):
        cos_a1, cos_a2, cos_b1, cos_b2 = self.get_cosine_dist(inp=inp, way=self.way, shot=self.shots[0])

        return cos_a1, cos_a2, cos_b1, cos_b2


