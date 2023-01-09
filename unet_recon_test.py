import pdb
import numpy as np
import os
import torchvision
import torch
from PIL import Image
import torch.nn.functional as F
from guided_diffusion.unet import UNetModel
from guided_diffusion.image_datasets import load_data
from logger import logger
import sys

#data_dir="/data/yzeng22/lsun_horse_adv256_clean_sample2048"
#data_dir="output_horse_sample2048/init/output"
data_dir=sys.argv[1]
out_dir = sys.argv[2]
#model_path="./output_autoenc_augblur_2_1.0_std0.2/net_40000.pth"
#model_cls_path="./output_autoenc_augblur_2_1.0_std0.2/net_cls_40000.pth"
model_path= "./imagenet_augblur_rotate_step20k_nonoise_tanh/net_124000.pth"
model_cls_path= "./imagenet_augblur_rotate_step20k_nonoise_tanh/net_cls_124000.pth"
dumt = 100
image_size = 256
batch_size = 8
device = torch.device("cuda:0")

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

print("load data")
data = load_data(
    data_dir=data_dir,
    batch_size=batch_size,
    image_size=image_size,
    class_cond=False,
    return_name=True,
    deterministic=True,
)

print("create models")
net = UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16, 32),
        dropout=0.1,
        channel_mult=(1,1,2,2,4,4),
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )
net.load_state_dict(torch.load(model_path))
net.to(device)
net.eval()

bdata = next(data, None)
while bdata is not None:
    sample, cond = bdata
    sample = sample.to(device)
    bsize,c,h,w = sample.size()
    t = torch.Tensor([dumt]*bsize).to(device)
    with torch.no_grad():
        recon = net(sample, t)
    #recon = torch.clamp(recon,-1,1).detach().cpu().numpy()/2+0.5
    recon = torch.tanh(recon).detach().cpu().numpy()/2+0.5
    recon = (recon*255).astype(np.uint8).transpose((0,2,3,1))
    bdata = next(data, None)
    for i, name in enumerate(cond['filename']):
        print(name)
        out = recon[i]
        Image.fromarray(out).save(f"{out_dir}/{name}")
