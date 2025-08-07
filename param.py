import torch
from torchsummary import summary
from unet import UNet
# from thop import profile


train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

# 定义模型
model =UNet(img_ch=3, output_ch=2)
model.to(device)

# 输出模型参数量大小

print(summary(model, input_size=(3, 512,512)))


# import torch
# from thop import profile
#
# # 将输入张量放置在 GPU 上
# input = torch.randn(1,3,224,224).to('cuda')
#
# # 创建模型，并将其放置在 GPU 上
# model = UNet(img_ch=3, output_ch=2).to('cuda')
#
# # 使用 thop.profile 函数计算模型的 FLOPs
# flops, params = profile(model, inputs=(input,))
#
# print(f"Model FLOPs: {flops}")
