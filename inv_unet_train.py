import pdb
import os
import argparse
import numpy as np
import random
import torchvision
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sys
sys.path.append("./guided-diffusion")
from guided_diffusion.unet import UNetModel
from guided_diffusion.image_datasets import load_data
import torchvision.transforms as transforms
from logger import logger

parser = argparse.ArgumentParser(description="training unet and classfier")

parser.add_argument("--data_dir", type=str)
parser.add_argument("--path_model", type=str)
parser.add_argument("--resume", type=int, required=False)
parser.add_argument("--log_dir", type=str, default="output/train")
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--train_steps", type=int, default=1e6)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--display_interval", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=2000)
args = parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#data_dir="/data/common/ILSVRC/Data/CLS-LOC/train"
#log_dir = "./imagenet_augblur_rotate_step20k_nonoise_tanh_cls0.01"
#resume=None#"./imagenet_augblur_2_1.0_std0.2_rotate/net_6000.pth"
#resume_cls=None#"./imagenet_augblur_2_1.0_std0.2_rotate/net_cls_6000.pth"
dumt = 100
image_size = args.image_size
batch_size = args.batch_size
train_steps = args.train_steps
log_interval = args.log_interval
display_interval = args.display_interval
save_interval = args.save_interval

device = torch.device("cuda:0")

writer = logger.Logger(args.log_dir)

with open(f"{args.log_dir}/args.txt","w") as f:
    for k,v in vars(args).items():
        f.write(f"{k}:{v}\n")

print("load data")
#transforms = transforms.Compose([
#    transforms.RandomApply([transforms.RandomAffine(15)], p=0.2),
#    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, saturation=0.2)], p=0.5),
#    transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3)])
data = load_data(
    data_dir=args.data_dir,
    batch_size=batch_size,
    image_size=image_size,
    class_cond=False,
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
print("model parameter number:")
print(count_parameters(net))
net.eval()

print("create patch models")
net_p = UNetModel(
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
print("model parameter number:")
print(count_parameters(net_p))

if args.resume is not None:
    path_ckpt = f"{args.log_dir}/net_p_{args.resume}.pth"
    if os.path.exists(path_ckpt):
        print(f"loading {path_ckpt}")
        net_p.load_state_dict(torch.load(path_ckpt))

net.load_state_dict(torch.load(args.path_model))
net.to(device)
net.eval()
net_p.to(device)
net_p.train()

list_param = list(net_p.parameters())
optim = torch.optim.Adam(
        list_param, 
        lr=1e-4)

if args.resume is not None:
    path_ckpt = f"{args.log_dir}/optim_{args.resume}.pth"
    if os.path.exists(path_ckpt):
        print(f"loading {path_ckpt}")
        optim.load_state_dict(torch.load(path_ckpt))

print("start training")

nstep=0 if args.resume is None else int(args.resume)
out0 = None
sample0 = None
while nstep<train_steps:
    sample, cond = next(data)
    sample = sample.to(device)
    bsize,c,h,w = sample.size()
    t = torch.Tensor([dumt]*bsize).to(device)
    with torch.no_grad():
        out = net(sample, t)
        out = torch.tanh(out)

    out_inv = net_p(out.detach(), t)
    out_inv = torch.tanh(out_inv)
    loss_recon = F.mse_loss(out_inv, sample)

    optim.zero_grad()
    loss_recon.backward()
    optim.step()
    if nstep % log_interval == 0:
        print(f"step {nstep} recon loss: {loss_recon.item()}")
        writer.add_scalar("recon loss", loss_recon.item(), nstep)
    if nstep % display_interval == 0:
        nd = 4
        disp = sample[:nd].transpose(0,1).reshape(-1, nd*h,w)
        writer.add_single_image("gt", disp/2+0.5)
        disp = out[:nd].transpose(0,1).reshape(-1, nd*h,w)
        writer.add_single_image("out", disp/2+0.5)
        disp = out_inv[:nd].transpose(0,1).reshape(-1, nd*h,w)
        writer.add_single_image("patch", disp/2+0.5)
        writer.write_html()
        print(f"step {nstep} saved images")
    if nstep % save_interval == 0:
        net_p.cpu()
        torch.save(net_p.state_dict(), f"{args.log_dir}/net_p_{nstep}.pth")
        net_p.to(device)
    nstep+=1
