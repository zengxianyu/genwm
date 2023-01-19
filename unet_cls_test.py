import pdb
import math
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
parser.add_argument("--num_labels", type=int, default=2)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--gt_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

num_bit = int(math.log(args.num_labels, 2))
assert 2**num_bit == args.num_labels

data_dir=args.data_dir
model_cls_path= args.model_path
dumt = 100
image_size = args.image_size
batch_size = args.batch_size
device = torch.device("cuda:0")

def load_data_from_path(load_data_dir):
    data = load_data(
        data_dir=load_data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=False,
        return_name=True,
        deterministic=True,
        return_loader=True
    )
    data = iter(data)
    return data

print("load data")

print("create classifier")
net_cls = torchvision.models.resnet34(pretrained=False)
net_cls.fc = torch.nn.Linear(net_cls.fc.in_features,num_bit)
torch.nn.init.kaiming_normal_(net_cls.fc.weight)
net_cls.load_state_dict(torch.load(model_cls_path))
net_cls.to(device)
net_cls.eval()

def predict(label_gt, data):
    num_correct = 0
    num_all = 0
    bdata = next(data, None)
    while bdata is not None:
        sample, cond = bdata
        sample = sample.to(device)
        bsize,c,h,w = sample.size()
        num_all += (bsize*num_bit)
        label = label_gt.expand(bsize,-1)
        t = torch.Tensor([dumt]*bsize).to(device)
        with torch.no_grad():
            pred = net_cls(sample)
            pred = (pred>0).float()
            num_correct += (pred==label).sum()
            batch_acc = (pred==label).sum().float()/(bsize*num_bit)
        print(f"batch acc: {batch_acc}")
        bdata = next(data, None)
    return num_correct, num_all

# fake images
num_correct = 0
num_all = 0
for rand_label in range(1, args.num_labels):
    if args.num_labels > 2:
        data_dir = f"{args.data_dir}/{rand_label}"
    else:
        data_dir = args.data_dir
    data = load_data_from_path(data_dir)
    #label_one = torch.Tensor([1]).to(device)[:,None]
    print(f"using label {rand_label}")
    label_one = "{0:b}".format(rand_label).zfill(num_bit)
    #print(f"using label {label_one}")
    label_one = [int(s) for s in label_one]
    label_one = torch.Tensor(label_one)[None].to(device)
    _num_correct, _num_all = predict(label_one, data)
    num_correct += _num_correct
    num_all += _num_all

# real images
data = load_data_from_path(args.gt_dir)
label_zero = torch.Tensor([0]).to(device)[:,None]
num_correct_gt, num_all_gt = predict(label_zero, data)
print(f"TotalAcc: {float(num_correct+num_correct_gt)/(num_all+num_all_gt)}")
