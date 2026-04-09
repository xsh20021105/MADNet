from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np
import math
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import sys
sys.path.append('/home/xsh/XSH/MADNet-master')

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def crop(d, g): 
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]
    d1 = d[:, :, int(math.floor((d_h - g_h) / 2.0)):int(math.floor((d_h - g_h) / 2.0)) + g_h,
         int(math.floor((d_w - g_w) / 2.0)):int(math.floor((d_w - g_w) / 2.0)) + g_w]
    return d1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ========================= DAH =========================
class DensityAwareHead(nn.Module):
    """
    输入: 单阶段特征  CxHxW
    输出: 3 通道密度图  3xHxW  (稀疏/正常/密集)
    """
    def __init__(self, in_channels):
        super(DensityAwareHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 3, 1, bias=True)   # 3 级密度
        )

    def forward(self, x):
        return self.conv(x)          # 3×H×W

# ========================= 密度加权特征融合 =========================
class DensityWeightedFusion(nn.Module):
    """
    1. 用 3 通道密度图经 Softmax 得像素级 3 级权重
    2. 将原特征复制 3 份，分别乘 3 级权重 → 3 组特征
    3. 每组特征再经 1x1 降回原始通道数 
    返回: 3 组特征(list),供后续 3 个头分别预测
    """
    def __init__(self, in_channels):
        super(DensityWeightedFusion, self).__init__()
        self.reduce = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])

    def forward(self, feat, density_logit):
        """
        feat:           CxHxW
        density_logit:  3xHxW
        """
        weight = F.softmax(density_logit, dim=1)        # 3×H×W  0~1
        feat3 = feat.unsqueeze(1).expand(-1, 3, -1, -1, -1)  # B×3×C×H×W
        weight = weight.unsqueeze(2)                    # B×3×1×H×W
        feat3 = feat3 * weight                          # B×3×C×H×W
        # 拆成 3 组
        feat_sparse, feat_normal, feat_dense = torch.unbind(feat3, dim=1)
        # 分别降维
        out = [self.reduce[i](feat) for i, feat in enumerate([feat_sparse, feat_normal, feat_dense])]
        return out
        
class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.seen = 0
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

#低秩空间注意力机制
class LRSA(nn.Module):
    """
    低秩空间自注意力：
    1. 把特征图在空间维压缩成 s×s 的“锚点” → 计算轻量 attention map
    2. 用线性插值上采样回原始分辨率，再加权回原始特征
    3. 整体复杂度 O(C·H·W + C·s²)，s=7 时显存可忽略
    """
    def __init__(self, in_channels, anchor=7):
        super().__init__()
        self.anchor = anchor
        mid = max(8, in_channels // 4)

        # Q/K/V 生成
        self.f_q = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.f_k = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.f_v = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # 输出变换
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.anchor

        # 1. 生成低分辨率 Q/K/V
        q = F.adaptive_avg_pool2d(self.f_q(x), (s, s))          # B×mid×s×s
        k = F.adaptive_avg_pool2d(self.f_k(x), (s, s))          # B×mid×s×s
        v = F.adaptive_avg_pool2d(self.f_v(x), (s, s))          # B×C ×s×s

        # 2. 拉平成 s² 个 token
        mid = q.size(1)
        q = q.view(B, mid, -1).permute(0, 2, 1)                 # B×s²×mid
        k = k.view(B, mid, -1)                                  # B×mid×s²
        v = v.view(B, C,  -1).permute(0, 2, 1)                  # B×s²×C

        # 3. 低秩 attention
        att = torch.bmm(q, k) * (mid ** -0.5)                   # B×s²×s²
        att = self.softmax(att)

        out = torch.bmm(att, v)                                 # B×s²×C
        out = out.permute(0, 2, 1).view(B, C, s, s)

        # 4. 插值回原始分辨率
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        out = self.out(out)
        return out

class DREFM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 局部
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        # 全局
        self.global_branch = LRSA(in_channels, anchor=7)

        # 融合
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1, bias=False),
            BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sigmoid()

    def forward(self, x):
        loc = self.local(x)
        glo = self.global_branch(x)
        fus = self.fuse(torch.cat([loc, glo], 1))
        return self.gate(fus) * fus

class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))
        

        self.stage2_head = DensityAwareHead(pre_stage_channels[0])   # 只用最高分辨率
        self.stage3_head = DensityAwareHead(pre_stage_channels[0])
        self.stage4_head = DensityAwareHead(pre_stage_channels[0])

        self.stage2_fuse = DensityWeightedFusion(pre_stage_channels[0])
        self.stage3_fuse = DensityWeightedFusion(pre_stage_channels[0])
        self.stage4_fuse = DensityWeightedFusion(pre_stage_channels[0])

        # 3 个轻量级密度头（1×1）
        self.head_sparse = nn.Conv2d(pre_stage_channels[0], 1, 1)
        self.head_normal = nn.Conv2d(pre_stage_channels[0], 1, 1)
        self.head_dense  = nn.Conv2d(pre_stage_channels[0], 1, 1)
        
        # 添加 DREFM 模块
        self.drefm = DREFM(last_inp_channels)

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(720, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0, bias=True),

        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        gt = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # ---------- Stage2 ----------
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            x_list.append(self.transition1[i](x) if self.transition1[i] else x)
        y_list = self.stage2(x_list)                    # list of multi-branch features
        h2 = y_list[0]                                  # 最高分辨率分支
        d2_logit = self.stage2_head(h2)                 # 3×H×W
        h2_sparse, h2_normal, h2_dense = self.stage2_fuse(h2, d2_logit)
        dens2 = [self.head_sparse(h2_sparse),
                self.head_normal(h2_normal),
                self.head_dense(h2_dense)]              # 3 张密度图

        # ---------- Stage3 ----------
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            x_list.append(self.transition2[i](y_list[-1]) if self.transition2[i] else y_list[i])
        y_list = self.stage3(x_list)
        h3 = y_list[0]
        d3_logit = self.stage3_head(h3)
        h3_sparse, h3_normal, h3_dense = self.stage3_fuse(h3, d3_logit)
        dens3 = [self.head_sparse(h3_sparse),
                self.head_normal(h3_normal),
                self.head_dense(h3_dense)]

        # ---------- Stage4 ----------
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            x_list.append(self.transition3[i](y_list[-1]) if self.transition3[i] else y_list[i])
        y_list = self.stage4(x_list)
        h4 = y_list[0]
        d4_logit = self.stage4_head(h4)
        h4_sparse, h4_normal, h4_dense = self.stage4_fuse(h4, d4_logit)
        dens4 = [self.head_sparse(h4_sparse),
                self.head_normal(h4_normal),
                self.head_dense(h4_dense)]

        # ---------- 最终融合 ----------
        # 把 3 组密度图取像素级最大
        final_dense = torch.max(torch.max(dens4[0], dens4[1]), dens4[2])
        # 继续走原来的 DREFM + last_layer
        x0_h, x0_w = h4.size(2), h4.size(3)
        x1 = F.upsample(y_list[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(y_list[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(y_list[3], size=(x0_h, x0_w), mode='bilinear')
        f = torch.cat([h4, x1, x2, x3], 1)
        f = self.drefm(f)
        x = self.last_layer(f)
        x = crop(x, gt)
        return x

        # 训练阶段返回多出口，推理阶段只用 x
        # if self.training:
        #     return {'final': x,
        #             'stage2': dens2,   # list of 3
        #             'stage3': dens3,
        #             'stage4': dens4}
        #     return x

    def init_weights(self, pretrained='', train=False):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if train==True:
            if os.path.isfile(pretrained):

                pretrained_dict = torch.load(pretrained)
                logger.info('=> loading pretrained model {}'.format(pretrained))
                model_dict = self.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict.keys()}
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict)

                print("load ImageNet pre_trained parameters for HR_Net")
            else:
                print('please check HRNET ImageNet pretrained model, the path ' + pretrained + ' is wrong')
                exit()


def get_seg_model(train=False):
    from Networks.HR_Net.default import _C as hr_config
    from Networks.HR_Net.default import update_config

    update_config(hr_config, '/home/xsh/XSH/MADNet-master/Networks/HR_Net/seg_hrnet_w48.yaml')
    model = HighResolutionNet(hr_config)
    from Networks.HR_Net.config import cfg
    
    # 添加 DREFM 模块初始化
    for m in model.modules():
        if isinstance(m, DREFM):
            for mm in m.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.normal_(mm.weight, std=0.001)
                elif isinstance(mm, nn.BatchNorm2d):
                    nn.init.constant_(mm.weight, 1)
                    nn.init.constant_(mm.bias, 0)
    
    model.init_weights(cfg.PRE_HR_WEIGHTS, train)
    return model

if __name__ == '__main__':
    from torchsummary import summary

    model = get_seg_model().cuda()
    print(model)
    summary(model, (3, 224, 224))

