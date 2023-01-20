import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import rich
console = rich.get_console()

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
    x1 = np.array(imw).astype(float)
    print(x1[0,0,:])
    x2 = np.array(im).astype(float)
    print(x2[0,0,:])
    diff = np.array(imw).astype(float) - np.array(im).astype(float)
    print(diff[0,0,:])
    console.print('diff.stat', diff.shape, diff.min(), diff.max(), diff.mean())
    kappa = np.linalg.norm(diff, ord=2, axis=2)
    console.print(f'[red]kappa.stat {kappa.shape} {kappa.min()} {kappa.max()} {kappa.mean()}')
    kappa = kappa / kappa.max()
    kappa = (kappa * 255).astype(np.uint8)
    imk = Image.fromarray(kappa)
    ax3.imshow(imk, cmap=__CMAP__)
    #plt.pcolormesh(np.flipud(kappa), cmap=__CMAP__)
    #plt.colorbar()
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
    del im, imw


def g_showtriple(ax1, px1, ax2, px2, ax3, px3,
                 *, ylabel=None):
    im1 = Image.open(px1)
    ax1.imshow(im1)
    ax1.set_ylabel(ylabel)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    im2 = Image.open(px2)
    ax2.imshow(im2)
    ax2.axis(False)
    im3 = Image.open(px3)
    ax3.imshow(im3)
    ax3.axis(False)

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

    fig = plt.figure(figsize=(14.4,20))

    ax1 = plt.subplot(5,3,1)
    ax2 = plt.subplot(5,3,2)
    ax3 = plt.subplot(5,3,3)
    g_showtriple(ax1, 'imgs/ffhq/sample_000020_ldm.png',
                 ax2, 'imgs/ffhq/sample_000109_ldm.png',
                 ax3, 'imgs/ffhq/sample_000109_ldm.png',
                 ylabel='LDM')

    ax4, ax5, ax6 = plt.subplot(5,3,4), plt.subplot(5,3,5), plt.subplot(5,3,6)
    g_showtriple(ax4, 'imgs/ffhq/gdcat_1012_1_adm.png',
                 ax5, 'imgs/ffhq/gdcat_1025_1_adm.png',
                 ax6, 'imgs/ffhq/gdcat_1027_1_adm.png',
                 ylabel='ADM')

    ax7 = plt.subplot(5,3,7)
    ax8 = plt.subplot(5,3,8)
    ax9 = plt.subplot(5,3,9)
    g_showtriple(ax7, 'imgs/ffhq/000057_stylegan2.png',
                 ax8, 'imgs/ffhq/000266_stylegan2.png',
                 ax9, 'imgs/ffhq/000932_stylegan2.png',
                 ylabel='StyleGAN2')

    ax10 = plt.subplot(5,3,10)
    ax11 = plt.subplot(5,3,11)
    ax12 = plt.subplot(5,3,12)
    g_showtriple(ax10, 'imgs/imagenet/ILSVRC2012_val_00016962.JPEG_ldm.png',
                 ax11, 'imgs/imagenet/ILSVRC2012_val_00023850.JPEG_ldm.png',
                 ax12, 'imgs/imagenet/ILSVRC2012_val_00040606.JPEG_ldm.png',
                 ylabel='LDM')

    ax13 = plt.subplot(5,3,13)
    ax14 = plt.subplot(5,3,14)
    ax15 = plt.subplot(5,3,15)
    g_showtriple(ax13, 'imgs/imagenet/ILSVRC2012_val_00000262.JPEG_adm.png',
                 ax14, 'imgs/imagenet/ILSVRC2012_val_00018250.JPEG_adm.png',
                 ax15, 'imgs/imagenet/ILSVRC2012_val_00041051.JPEG_adm.png',
                 ylabel='ADM')


    plt.tight_layout()

if ag.save is None:
    plt.show()
else:
    plt.savefig(ag.save)

