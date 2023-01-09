import pdb
import torchvision
import torch
import torch.nn.functional as F
from guided_diffusion.unet import UNetModel
from guided_diffusion.image_datasets import load_data
from logger import logger
import sys

data_dir=sys.argv[1]
#model_path="./output_autoenc_augblur_2_1.0_std0.2/net_40000.pth"
#model_cls_path="./output_autoenc_augblur_2_1.0_std0.2/net_cls_40000.pth"
model_path= "./imagenet_augblur_rotate_step20k_nonoise_tanh/net_124000.pth"
model_cls_path= "./imagenet_augblur_rotate_step20k_nonoise_tanh/net_cls_124000.pth"
dumt = 100
image_size = 256
batch_size = 16
device = torch.device("cuda:0")

print("load data")
data = load_data(
    data_dir=data_dir,
    batch_size=batch_size,
    image_size=image_size,
    class_cond=False,
    return_name=True,
    deterministic=True,
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
print("TotalAcc:")
print(float(num_correct)/num_all)
