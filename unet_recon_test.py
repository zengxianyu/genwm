import pdb
import math
import argparse
import numpy as np
import os
import torchvision
import torch
from PIL import Image
import torch.nn.functional as F
import sys
sys.path.append("./guided-diffusion")
from guided_diffusion.unet import UNetModel
from guided_diffusion.image_datasets import load_data
from logger import logger

parser = argparse.ArgumentParser(description="training unet and classfier")
parser.add_argument("--use_noise", action='store_true')
parser.add_argument("--noise_scale", type=float, default=0.2)
parser.add_argument("--num_labels", type=int, default=2)
parser.add_argument("--return_prefix", action='store_true')
parser.add_argument("--data_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

num_bit = int(math.log(args.num_labels, 2))
assert 2**num_bit == args.num_labels

data_dir=args.data_dir
out_dir = args.log_dir
model_path= args.model_path
dumt = 100
image_size = args.image_size
batch_size = args.batch_size

device = torch.device("cuda:0")

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

print("load data")
dataloader = load_data(
    data_dir=data_dir,
    batch_size=batch_size,
    image_size=image_size,
    class_cond=False,
    return_name=True,
    return_prefix=args.return_prefix,
    deterministic=True,
    return_loader=True,
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
        num_classes=-num_bit if args.num_labels>2 else None,
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

for rand_label in range(1, args.num_labels):
    idx = 0
    data = iter(dataloader)
    bdata = next(data, None)
    while bdata is not None:
        sample, cond = bdata
        sample = sample.to(device)
        noise = torch.randn_like(sample)*args.noise_scale
        bsize,c,h,w = sample.size()

        print(f"using label {rand_label}")
        label_one = "{0:b}".format(rand_label).zfill(num_bit)
        print(f"using label {label_one}")
        label_one = [int(s) for s in label_one]
        label_one = torch.Tensor(label_one)[None].to(device)

        if args.num_labels > 2:
            out_dir = f"{args.log_dir}/{rand_label}"
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            y_in = label_one.expand(bsize,-1)
        else:
            out_dir = args.log_dir
            y_in = None

        t = torch.Tensor([dumt]*bsize).to(device)
        with torch.no_grad():
            #recon = net(sample, t, y=y_in)
            if args.use_noise:
                recon = net(sample+noise, t, y=y_in)
            else:
                recon = net(sample, t, y=y_in)
        #recon = torch.clamp(recon,-1,1).detach().cpu().numpy()/2+0.5
        recon = torch.tanh(recon).detach().cpu().numpy()/2+0.5
        recon = (recon*255).astype(np.uint8).transpose((0,2,3,1))
        bdata = next(data, None)
        for i, name in enumerate(cond['filename']):
            print(f"{idx}/{len(data)}: {name}")
            if 'prefix' in cond:
                prefix = cond['prefix']
                if not os.path.exists(f"{out_dir}/{prefix[i]}"):
                    os.mkdir(f"{out_dir}/{prefix[i]}")
                full_path = f"{out_dir}/{prefix[i]}"
            else:
                full_path = out_dir
            out = recon[i]
            Image.fromarray(out).save(f"{full_path}/{name}")
        idx += 1
