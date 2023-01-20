import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# configs
__FONTSIZE__ = 32
font = {'family': 'monospace', 'weight': 'normal', 'size': __FONTSIZE__}
matplotlib.rc('font', **font)

# arguments
ag = argparse.ArgumentParser()
ag.add_argument('-s', '--save', type=str, default=None)
ag.add_argument('which', type=str, choices=['w', 'g'])
ag = ag.parse_args()

# helpers
def w_showtriple(ax1, gt, ax2, w, ax3, *, ylabel=None, title=None):
    __CMAP__ = 'RdPu'
    # 1
    im = Image.open(gt)
    ax1.imshow(im)
    if title is not None:
        ax1.set_title(title, fontsize=__FONTSIZE__)
    if ylabel is not None:
        ax1.set_ylabel(ylabel[0], fontsize=__FONTSIZE__)
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
    if ylabel is not None:
        ax2.set_ylabel(ylabel[1], fontsize=__FONTSIZE__)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    # 3
    kappa = np.linalg.norm(np.array(imw) - np.array(im), axis=2)
    kappa = kappa / kappa.max()
    kappa = (kappa * 255).astype(np.uint8)
    imk = Image.fromarray(kappa)
    ax3.imshow(imk, cmap=__CMAP__)
    if ylabel is not None:
        ax3.set_ylabel(ylabel[2], fontsize=__FONTSIZE__)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)


# draw
if ag.which == 'w':
    # stage 1
    fig = plt.figure(figsize=(20, 14.4))

    ax1 = plt.subplot(3,4,1)
    ax2 = plt.subplot(3,4,5)
    ax3 = plt.subplot(3,4,9)
    w_showtriple(ax1, 'imgs/ffhq/01627_GT.png',
                 ax2, 'imgs/ffhq/01627_W.png',
                 ax3,
                 ylabel=['Ground Truth $x$',
                        'Watermarked $\\hat{x}$',
                        'Watermark $\\kappa$',],
                 title='FFHQ')
    ax4 = plt.subplot(3,4,2)
    ax5 = plt.subplot(3,4,6)
    ax6 = plt.subplot(3,4,10)
    w_showtriple(ax4, 'imgs/ffhq/28170_GT.png',
                 ax5, 'imgs/ffhq/28170_W.png',
                 ax6,
                 title='FFHQ')
    ax7 = plt.subplot(3,4,3)
    ax8 = plt.subplot(3,4,7)
    ax9 = plt.subplot(3,4,11)
    w_showtriple(ax7, 'imgs/imagenet/ILSVRC2012_val_00025488_GT.JPEG',
                 ax8, 'imgs/imagenet/ILSVRC2012_val_00025488_W.JPEG',
                 ax9,
                 title='ImageNet')
    ax10 = plt.subplot(3,4,4)
    ax11 = plt.subplot(3,4,8)
    ax12 = plt.subplot(3,4,12)
    w_showtriple(ax10, 'imgs/imagenet/ILSVRC2012_val_00027351_GT.JPEG',
                 ax11, 'imgs/imagenet/ILSVRC2012_val_00027351_W.JPEG',
                 ax12,
                 title='ImageNet')
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

