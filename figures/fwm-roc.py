import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import rich
console = rich.get_console()

# config
font = {'family': 'monospace', 'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)

# argparse
ag = argparse.ArgumentParser()
ag.add_argument('--save', type=str, default=None)
ag = ag.parse_args()
console.print(ag)

fig = plt.figure(figsize=(19.2, 4.8))

yt1 = np.loadtxt('data/ffhq_1bit_0.05/120000.ytrue.txt')
ys1 = np.loadtxt('data/ffhq_1bit_0.05/120000.yscores.txt')
yt1m = np.loadtxt('data/ffhq_fw/120000_inv.ytrue.txt')
ys1m = np.loadtxt('data/ffhq_fw/120000_inv.yscores.txt')
ax1 = plt.subplot(1,4,1)
fpr1, tpr1, _ = roc_curve(yt1, ys1, pos_label=1)
plt.plot(fpr1, tpr1, marker='o', label='$F(W(x))$')
fpr1m, tpr1m, _ = roc_curve(yt1m, ys1m, pos_label=1)
plt.plot(fpr1m, tpr1m, marker='.', label='$F(M(W(x)))$')
plt.plot([0, 1], [0, 1], 'k:', label='_nolegend_')
ax1.legend()
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('w/o $e$, w/o $L_{aux}$')

yt2 = np.loadtxt('data/ffhq_fwm/64000.ytrue.txt')
ys2 = np.loadtxt('data/ffhq_fwm/64000.yscores.txt')
yt2m = np.loadtxt('data/ffhq_fwm/64000_inv.ytrue.txt')
ys2m = np.loadtxt('data/ffhq_fwm/64000_inv.yscores.txt')
ax2 = plt.subplot(1,4,2)
fpr2, tpr2, _ = roc_curve(yt2, ys2, pos_label=1)
plt.plot(fpr2, tpr2, marker='o', label='$F(W(x))$')
fpr2m, tpr2m, _ = roc_curve(yt2m, ys2m, pos_label=1)
plt.plot(fpr2m, tpr2m, marker='.', label='$F(M(W(x)))$')
plt.plot([0, 1], [0, 1], 'k:', label='_nolegend_')
ax2.legend()
ax2.set_xlabel('False Positive Rate')
ax2.set_title('w/o $e$, w/ $L_{aux}$')

yt3 = np.loadtxt('data/ffhq_noise_fw/noise_scale0.5.ytrue.txt')
ys3 = np.loadtxt('data/ffhq_noise_fw/noise_scale0.5.yscores.txt')
yt3m = np.loadtxt('data/ffhq_noise_fw/noise_scale0.5_inv.ytrue.txt')
ys3m = np.loadtxt('data/ffhq_noise_fw/noise_scale0.5_inv.yscores.txt')
ax3 = plt.subplot(1,4,3)
fpr3, tpr3, _ = roc_curve(yt3, ys3, pos_label=1)
plt.plot(fpr3, tpr3, marker='o', label='$F(W(x))$')
fpr3m, tpr3m, _ = roc_curve(yt3m, ys3m, pos_label=1)
plt.plot(fpr3m, tpr3m, marker='.', label='$F(M(W(x)))$')
plt.plot([0, 1], [0, 1], 'k:', label='_nolegend_')
ax3.legend()
ax3.set_xlabel('False Positive Rate')
ax3.set_title('w/ $e$, w/o $L_{aux}$')


yt4 = np.loadtxt('data/ffhq_noise_fwm/94000.ytrue.txt')
ys4 = np.loadtxt('data/ffhq_noise_fwm/94000.yscores.txt')
yt4m = np.loadtxt('data/ffhq_noise_fwm/94000_inv.yture.txt')
ys4m = np.loadtxt('data/ffhq_noise_fwm/94000_inv.yscores.txt')
ax4 = plt.subplot(1,4,4)
fpr4, tpr4, _ = roc_curve(yt4, ys4, pos_label=1)
plt.plot(fpr4, tpr4, marker='o', label='$F(W(x))$')
fpr4m, tpr4m, _ = roc_curve(yt4m, ys4m, pos_label=1)
plt.plot(fpr4m, tpr4m, marker='.', label='$F(M(W(x)))$', linewidth=3)
plt.plot([0, 1], [0, 1], 'k:', label='_nolegend_')
ax4.legend()
ax4.set_xlabel('False Positive Rate')
ax4.set_title('w/ $e$, w/ $L_{aux}$')


plt.tight_layout()
if ag.save is None:
    plt.show()
else:
    plt.savefig(ag.save)
