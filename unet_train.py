import pdb
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
parser.add_argument("--log_dir", type=str, default="output/train")
parser.add_argument("--quant", action='store_true')
parser.add_argument("--resume", type=str, required=False)
parser.add_argument("--resume_cls", type=str, required=False)
parser.add_argument("--w_cls0", type=float, default=0.1)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--cls_start", type=int, default=20000)
parser.add_argument("--train_steps", type=int, default=1e6)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--display_interval", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=2000)
parser.add_argument("--use_noise", action="store_true")
args = parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#data_dir="/data/common/ILSVRC/Data/CLS-LOC/train"
#log_dir = "./imagenet_augblur_rotate_step20k_nonoise_tanh_cls0.01"
#resume=None#"./imagenet_augblur_2_1.0_std0.2_rotate/net_6000.pth"
#resume_cls=None#"./imagenet_augblur_2_1.0_std0.2_rotate/net_cls_6000.pth"
dumt = 100
w_cls0 = args.w_cls0
image_size = args.image_size
batch_size = args.batch_size
cls_start=args.cls_start
train_steps = args.train_steps
log_interval = args.log_interval
display_interval = args.display_interval
save_interval = args.save_interval

device = torch.device("cuda:0")

writer = logger.Logger(args.log_dir)

with open(f"{args.log_dir}/args.txt","w") as f:
    for k,v in vars(args).items():
        f.write(f"{k}:{v}\n")

class GaussianBlur:
    def __init__(self, size=5, device=torch.device('cpu'), channels=3, sigmas=[1]):
        self.kernels = [self.gaussian_kernel(size, device, channels, sigma) \
                for sigma in sigmas]
    def blur(self, x):
        kernel = random.choice(self.kernels)
        return self.gaussian_conv2d(x, kernel)
    def gaussian_kernel(self, size=5, device=torch.device('cpu'), channels=3, sigma=1, dtype=torch.float):
        # Create Gaussian Kernel. In Numpy
        interval  = (2*sigma +1)/(size)
        ax = np.linspace(-(size - 1)/ 2., (size-1)/2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx)+ np.square(yy)) / np.square(sigma))
        kernel /= np.sum(kernel)
        # Change kernel to PyTorch. reshapes to (channels, 1, size, size)
        kernel_tensor = torch.as_tensor(kernel, dtype=dtype)
        kernel_tensor = kernel_tensor.repeat(channels, 1 , 1, 1).to(device)
        return kernel_tensor
    def gaussian_conv2d(self, x, g_kernel, dtype=torch.float):
        #Assumes input of x is of shape: (minibatch, depth, height, width)
        #Infer depth automatically based on the shape
        channels = g_kernel.shape[0]
        padding = g_kernel.shape[-1] // 2 # Kernel size needs to be odd number
        if len(x.shape) != 4:
            raise IndexError('Expected input tensor to be of shape: (batch, depth, height, width) but got: ' + str(x.shape))
        y = F.conv2d(x, weight=g_kernel, stride=1, padding=padding, groups=channels)
        return y

class RandomRotation:
    def __init__(self, angles):
        self.angles=angles
    def rotate(self, x):
        angle = random.choice(self.angles)
        if angle == 0:
            return x
        return TF.rotate(x, angle)

rotation = RandomRotation(angles=list(np.linspace(-30, 30, 61)))
blur = GaussianBlur(device=device, sigmas=np.linspace(0.01,1,10))
flip = torchvision.transforms.RandomHorizontalFlip(0.5)

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
if args.resume is not None:
    print(f"resume {args.resume}")
    net.load_state_dict(torch.load(args.resume))
net.to(device)
net.train()
print("model parameter number:")
print(count_parameters(net))

print("create classifier")
net_cls = torchvision.models.resnet34(pretrained=False)
net_cls.fc = torch.nn.Linear(net_cls.fc.in_features,1)
torch.nn.init.kaiming_normal_(net_cls.fc.weight)
if args.resume_cls is not None:
    print(f"resume cls {args.resume_cls}")
    net_cls.load_state_dict(torch.load(args.resume_cls))
net_cls.to(device)
net_cls.train()
print("classifier parameter number:")
print(count_parameters(net_cls))

print("start training")
optim = torch.optim.Adam(
        list(net.parameters())+list(net_cls.parameters()), 
        lr=1e-4)

nstep=0
label_one = torch.Tensor([1]).to(device)[:,None]
label_zero = torch.Tensor([0]).to(device)[:,None]
out0 = None
sample0 = None
while nstep<train_steps:
    sample, cond = next(data)
    sample = sample.to(device)
    noise = torch.randn_like(sample)*0.2
    flag = random.randint(0,1) if args.use_noise else 0
    if flag:
        sample = sample+noise
    bsize,c,h,w = sample.size()
    t = torch.Tensor([dumt]*bsize).to(device)
    out = net(sample, t)
    if not args.use_noise:
        out = torch.tanh(out)
    if args.quant:
        out_clamp = torch.clamp(out, -1, 1)/2+0.5
        out_q = (out_clamp*255).byte().float()/255*2-1
        err = out_q-out
        out = out+err.detach()
        #loss_q = F.l1_loss(out_clamp, out_q.detach())
    #else:
    #    loss_q = 0
    # classification loss
    # reconstruction loss
    loss_recon = F.mse_loss(out, sample)
    if out0 is None or sample0 is None or not flag:
        out0 = out
        sample0 = sample
    # augmentation
    out = rotation.rotate(out)
    sample = rotation.rotate(sample)
    out = blur.blur(out)
    sample = blur.blur(sample)
    out = flip(out)
    sample = flip(sample)

    pred = net_cls(torch.cat([out, sample],0))
    label = torch.cat([label_one.expand(bsize,-1), label_zero.expand(bsize,-1)],0)
    loss_cls = F.binary_cross_entropy_with_logits(pred, label)
    if nstep<cls_start:
        w_cls = float(nstep)/cls_start * w_cls0
    else:
        w_cls = w_cls0
    print(f"weight cls: {w_cls}")

    loss = loss_cls*w_cls+loss_recon#+loss_q*w_cls

    optim.zero_grad()
    loss.backward()
    optim.step()
    if nstep % log_interval == 0:
        print(f"step {nstep} recon loss: {loss_recon.item()}")
        print(f"step {nstep} cls loss: {loss_cls.item()}")
        writer.add_scalar("recon loss", loss_recon.item(), nstep)
        writer.add_scalar("cls loss", loss_cls.item(), nstep)
    if nstep % display_interval == 0:
        nd = 4
        disp = sample0[:nd].transpose(0,1).reshape(-1, nd*h,w)
        writer.add_single_image("gt", disp/2+0.5)
        disp = out0[:nd].transpose(0,1).reshape(-1, nd*h,w)
        writer.add_single_image("result", disp/2+0.5)
        writer.write_html()
        print(f"step {nstep} saved images")
    if nstep % save_interval == 0:
        net_cls.cpu()
        net.cpu()
        torch.save(net.state_dict(), f"{args.log_dir}/net_{nstep}.pth")
        torch.save(net_cls.state_dict(), f"{args.log_dir}/net_cls_{nstep}.pth")
        net.to(device)
        net_cls.to(device)
    nstep+=1
