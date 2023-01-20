import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# configs
font = {'family': 'monospace', 'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)

# arguments
ag = argparse.ArgumentParser()
ag.add_argument('-s', '--save', type=str, default=None)
ag.add_argument('which', type=str, choices=['w', 'g'])
ag = ag.parse_args()

# helpers
def w_showtriple(ax1, gt, ax2, w, ax3, *, ylabel=None, title=False):
    __CMAP__ = 'RdPu'
    # 1
    im = Image.open(gt)
    ax1.imshow(im)
    if title:
        ax1.set_title('Ground Truth $x$')
    if ylabel is not None:
        ax1.set_ylabel(ylabel)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    # 2
    imw = Image.open(w)
    ax2.imshow(imw)
    ax2.axis(False)
    if title:
        ax2.set_title('Watermarked $\\hat{x}$')
    # 3
    kappa = np.linalg.norm(np.array(imw) - np.array(im), axis=2)
    kappa = kappa / kappa.max()
    kappa = (kappa * 255).astype(np.uint8)
    imk = Image.fromarray(kappa)
    ax3.imshow(imk, cmap=__CMAP__)
    #plt.colorbar(ax=ax3)
    ax3.axis(False)
    if title:
        ax3.set_title('Normalized $\\kappa$')


# draw
if ag.which == 'w':
    # stage 1
    fig = plt.figure(figsize=(14.4,20))

    ax1 = plt.subplot(4,3,1)
    ax2 = plt.subplot(4,3,2)
    ax3 = plt.subplot(4,3,3)
    w_showtriple(ax1, 'imgs/ffhq/01627_GT.png',
                 ax2, 'imgs/ffhq/01627_W.png',
                 ax3,
                 ylabel='FFHQ',
                 title=True)

    ax4 = plt.subplot(4,3,4)
    ax5 = plt.subplot(4,3,5)
    ax6 = plt.subplot(4,3,6)
    w_showtriple(ax4, 'imgs/ffhq/28170_GT.png',
                 ax5, 'imgs/ffhq/28170_W.png',
                 ax6,
                 ylabel='FFHQ')

    ax7 = plt.subplot(4,3,7)
    ax8 = plt.subplot(4,3,8)
    ax9 = plt.subplot(4,3,9)
    w_showtriple(ax7, 'imgs/imagenet/ILSVRC2012_val_00025488_GT.JPEG',
                 ax8, 'imgs/imagenet/ILSVRC2012_val_00025488_W.JPEG',
                 ax9,
                 ylabel='ImageNet')

    ax10 = plt.subplot(4,3,10)
    ax11 = plt.subplot(4,3,11)
    ax12 = plt.subplot(4,3,12)
    w_showtriple(ax10, 'imgs/imagenet/ILSVRC2012_val_00027351_GT.JPEG',
                 ax11, 'imgs/imagenet/ILSVRC2012_val_00027351_W.JPEG',
                 ax12,
                 ylabel='ImageNet')

    plt.tight_layout()

elif ag.which == 'g':

    raise NotImplementedError

if ag.save is None:
    plt.show()
else:
    plt.savefig(ag.save)

#figures/imgs/ffhq/000057_stylegan2.png
#figures/imgs/ffhq/000266_stylegan2.png
#figures/imgs/ffhq/000932_stylegan2.png
#figures/imgs/ffhq/gdcat_1012_1_adm.png
#figures/imgs/ffhq/gdcat_1025_1_adm.png
#figures/imgs/ffhq/gdcat_1027_1_adm.png
#figures/imgs/ffhq/sample_000020_ldm.png
#figures/imgs/ffhq/sample_000109_ldm.png
#figures/imgs/imagenet/ILSVRC2012_val_00000262.JPEG_adm.png
#figures/imgs/imagenet/ILSVRC2012_val_00016962.JPEG_ldm.png
#figures/imgs/imagenet/ILSVRC2012_val_00018250.JPEG_adm.png
#figures/imgs/imagenet/ILSVRC2012_val_00023850.JPEG_ldm.png
#figures/imgs/imagenet/ILSVRC2012_val_00040606.JPEG_ldm.png
#figures/imgs/imagenet/ILSVRC2012_val_00041051.JPEG_adm.png

