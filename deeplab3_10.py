import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
from torchstat import stat
import torchvision.models as models



class WeightNorm(nn.Module):
    def __init__(self, num_features):
        super(WeightNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1))

    def forward(self, x):
        return self.weight * x + self.bias

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(ResidualBlock, self).__init__()

        # 第一个dilatedConv层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation_rate,  padding=(kernel_size-1)//2 * dilation_rate)
        self.norm1 = nn.utils.weight_norm(nn.BatchNorm2d(out_channels))
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.5)

        # 第二个dilatedConv层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, dilation=dilation_rate,  padding=(kernel_size-1)//2 * dilation_rate)
        self.norm2 = nn.utils.weight_norm(nn.BatchNorm2d(out_channels))
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(0.5)

        # 1x1卷积层，用于调整通道数
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.elu1(x)
        x = self.dropout1(x)
        # print(x.size())

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.elu2(x)
        x = self.dropout2(x)
        # print(x.size())

        # 将输入x通过1x1卷积调整通道数
        residual = self.conv1x1(residual)
        # print(residual.size())

        # 将调整通道数后的输入x与残差块的输出相加
        x = x + residual

        return x





class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out




class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x

    # -----------------------------------------#


#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.ECA2 = ECA_block(channel=1280)  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!添加了这里!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        feature_cat = self.ECA2(feature_cat)    #!!!!!!!!!!!!!!!!!!!!!!!!!!添加了这里!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print(feature_cat.shape)
        result = self.conv_cat(feature_cat)
        return result



class DeepLab(nn.Module):
    def __init__(self, input_channel, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=8):
        super(DeepLab, self).__init__()  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!修改了downsample_factor原来是16!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.num_classes = num_classes
        self.input_channel = input_channel
        self.ECA1= ECA_block(channel = 24)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!添加了这里!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320   #2048
            low_level_channels = 24   #256
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#

        # self.Dilated_Conv = ResidualBlock(48, 48, kernel_size=5, dilation_rate=4)  #!!!!!!!!!!!!!!!!!!!!!!!!!!!添加了这里!!!!!!!!!!!!!!!!!!!!!!!


        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        # self.generate_heatmap(x)
        low_level_features = self.ECA1(low_level_features)   #!!!!!!!!!!!!!!!!!!!!!!!!!!!添加了这里!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.generate_heatmap(x)

        low_level_features = self.shortcut_conv(low_level_features)
        # self.generate_heatmap(x)

        # low_level_features = self.Dilated_Conv(low_level_features)   #!!!!!!!!!!!!!!!!!!!!!!!!!!!添加了这里!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        self.generate_heatmap(x)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        self.generate_heatmap(x)
        x = self.cls_conv(x)
        self.generate_heatmap(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        self.generate_heatmap(x)
        return x

    def generate_heatmap(self, feature_map):
        """
        Generate and display a heatmap by combining all channels from the given feature map (d2).
        :param feature_map: The feature map (d2) to visualize.
        """
        # 将特征图转换为 numpy 数组并做最大最小归一化
        feature_map = feature_map.squeeze().cpu().detach().numpy()  # 移除批次维度并转换为 NumPy 数组

        # 如果有多个通道，逐个通道叠加
        if len(feature_map.shape) == 3:  # 通道数大于1
            num_channels = feature_map.shape[0]

            # 初始化一个全零的热图
            combined_heatmap = np.zeros_like(feature_map[0])  # 选择第一个通道的形状作为目标大小

            # 将所有通道的特征图叠加
            for i in range(num_channels):
                channel_map = feature_map[i]
                # 对每个通道进行归一化 (最小最大归一化)
                channel_map = (channel_map - np.min(channel_map)) / (np.max(channel_map) - np.min(channel_map) + 1e-8)
                combined_heatmap += channel_map  # 叠加到一起

            # 归一化叠加后的热图
            combined_heatmap = (combined_heatmap - np.min(combined_heatmap)) / (
                    np.max(combined_heatmap) - np.min(combined_heatmap) + 1e-8)

        else:
            # 如果只有一个通道，则直接使用它
            combined_heatmap = feature_map

        # 使用 matplotlib 生成热图
        plt.imshow(combined_heatmap, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.title('')
        plt.show()



# model = DeepLab(input_channel=3,num_classes=2, backbone="mobilenet", pretrained=True, downsample_factor=16)
# stat(model, (3, 512, 512))
