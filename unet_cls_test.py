import pdb
import argparse
import torchvision
import torch
import torch.nn.functional as F
import sys
sys.path.append("./guided-diffusion")
from guided_diffusion.unet import UNetModel
from guided_diffusion.image_datasets import load_data
from logger import logger

parser = argparse.ArgumentParser(description="training unet and classfier")
parser.add_argument("--data_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

data_dir=args.data_dir
model_cls_path= args.model_path
dumt = 100
image_size = args.image_size
batch_size = args.batch_size
device = torch.device("cuda:0")

print("load data")
data = load_data(
    data_dir=data_dir,
    batch_size=batch_size,
    image_size=image_size,
    class_cond=False,
    return_name=True,
    deterministic=True,
    return_loader=True
)

print("create classifier")
net_cls = torchvision.models.resnet34(pretrained=False)
net_cls.fc = torch.nn.Linear(net_cls.fc.in_features,1)
torch.nn.init.kaiming_normal_(net_cls.fc.weight)
net_cls.load_state_dict(torch.load(model_cls_path))
net_cls.to(device)
net_cls.eval()

label_one = torch.Tensor([1]).to(device)[:,None]
num_correct = 0
num_all = 0
data = iter(data)
bdata = next(data, None)
while bdata is not None:
    sample, cond = bdata
    sample = sample.to(device)
    bsize,c,h,w = sample.size()
    num_all += (bsize)
    label = label_one.expand(bsize,-1)
    t = torch.Tensor([dumt]*bsize).to(device)
    with torch.no_grad():
        pred = net_cls(sample)
        pred = (pred>0).float()
        num_correct += (pred==label).sum()
        batch_acc = (pred==label).sum().float()/(bsize)
    print(f"batch acc: {batch_acc}")
    bdata = next(data, None)
print(f"TotalAcc: {float(num_correct)/num_all}")
