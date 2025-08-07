import torch
import torch.nn as nn
import torch.nn.functional as F

# BP-Net和TCN
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
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_rate,  padding=(kernel_size-1)//2 * dilation_rate)
        self.norm1 = nn.utils.weight_norm(nn.BatchNorm1d(out_channels))
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.5)

        # 第二个dilatedConv层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation_rate,  padding=(kernel_size-1)//2 * dilation_rate)
        self.norm2 = nn.utils.weight_norm(nn.BatchNorm1d(out_channels))
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(0.5)

        # 1x1卷积层，用于调整通道数
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

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

# ####################################### BP-Net ##############################################
class PPGModel(nn.Module):
    def __init__(self):
        super(PPGModel, self).__init__()
        self.block1 = ResidualBlock(32, 32, kernel_size=5, dilation_rate=1)
        self.block2 = ResidualBlock(32, 32, kernel_size=5, dilation_rate=2)
        self.block3 = ResidualBlock(32, 64, kernel_size=5, dilation_rate=4)
        self.block4 = ResidualBlock(64, 64, kernel_size=5, dilation_rate=8)
        self.block5 = ResidualBlock(64, 128, kernel_size=5, dilation_rate=16)
        self.block6 = ResidualBlock(128, 256, kernel_size=5, dilation_rate=32)

        self.conv1x1 = nn.Conv1d(1, 32, kernel_size=1)
        # 1x1 Convolution with ELU
        self.conv1x1a = nn.Conv1d(256, 256, kernel_size=1)
        self.conv1x1b = nn.Conv1d(256, 2, kernel_size=1)

        self.elu = nn.ELU()

        # Fully Connected Layer
        # self.fc = nn.Linear(256, 2)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.size())
        x = self.conv1x1(x)
        x = self.block1(x)
        # print(x.size())
        x = self.block2(x)
        # print(x.size())
        x = self.block3(x)
        # print(x.size())
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        print("1", x.size())

        # 1x1 Convolution with ELU
        x = self.elu(self.conv1x1a(x))
        x = self.elu(self.conv1x1b(x))

        # 自适应平均池化
        x = F.adaptive_avg_pool1d(x, 1).squeeze(dim=2)
        print("2", x.size())

        # Fully Connected Layer
        # x = self.fc(x)
        return x


# ####################################### TCN ##############################################

# class TCN(nn.Module):
#     def __init__(self):
#         super(TCN, self).__init__()
#         self.block1 = ResidualBlock(32, 32, kernel_size=2, dilation_rate=2)
#         self.block2 = ResidualBlock(32, 32, kernel_size=2, dilation_rate=2)
#         self.block3 = ResidualBlock(32, 64, kernel_size=2, dilation_rate=4)
#         self.block4 = ResidualBlock(64, 64, kernel_size=2, dilation_rate=8)
#         self.block5 = ResidualBlock(64, 64, kernel_size=2, dilation_rate=16)
#         self.block6 = ResidualBlock(64, 128, kernel_size=2, dilation_rate=32)
#         self.block7 = ResidualBlock(128, 128, kernel_size=2, dilation_rate=64)
#         self.block8 = ResidualBlock(128, 128, kernel_size=2, dilation_rate=128)
#         self.block9 = ResidualBlock(128, 256, kernel_size=2, dilation_rate=256)
#
#         self.conv1x1 = nn.Conv1d(1, 32, kernel_size=1)
#         # 1x1 Convolution with ELU
#         self.conv1x1a = nn.Conv1d(256, 256, kernel_size=1)
#         self.conv1x1b = nn.Conv1d(256, 2, kernel_size=1)
#
#         self.relu = ReLU()
#
#         # Fully Connected Layer
#         # self.fc = nn.Linear(256, 2)
#         # self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.conv1x1(x)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.block6(x)
#         x = self.block7(x)
#         x = self.block8(x)
#         x = self.block9(x)
#
#         # 1x1 Convolution with ELU
#         x = self.relu(self.conv1x1a(x))
#         x = self.relu(self.conv1x1b(x))
#
#         # 自适应平均池化
#         x = F.adaptive_avg_pool1d(x, 1).squeeze(dim=2)
#
#         # Fully Connected Layer
#         # x = self.fc(x)
#         return x

