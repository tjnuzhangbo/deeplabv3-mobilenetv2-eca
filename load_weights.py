import os
import torch
import torch.nn as nn
from unet import UNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weight_path = r"D:\赵\Pytorch-UNet-3.0(yellow) - 副本\checkpoints\checkpoint_epoch200.pth"
    assert os.path.exists(model_weight_path),"file{} does not exists.".format(model_weight_path)
    net = UNet(img_ch=3, output_ch=2)
    net_weight = net.state_dict()
    pre_weights = torch.load(model_weight_path,map_location = device)
    del_key = []
    for key,_ in pre_weights.items():
        if "fc" in key:
            del_key.append(key)
    for key in del_key:
        del pre_weights[key]
    missing_key,unexpected_key = net.load_state_dict(pre_weights, strict = False)
    print("[missing_key:]",*missing_key,sep="\n")
    print("[unexpected_key:]", *unexpected_key, sep="\n")

if __name__=='__main__':
    main()