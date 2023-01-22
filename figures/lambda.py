import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

# configs
font = {'family': 'monospace', 'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)

# arguments
ag = argparse.ArgumentParser()
ag.add_argument('-s', '--save', type=str, default=None)
ag = ag.parse_args()

# data
lambdas = np.array([0.00001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0])
psnrs = np.array([63.0, 54.9, 52.7, 52.1, 51.4, 42.8, 37.5])
#psnrs = np.nan_to_num(psnrs, 0.0)
fids = np.array([0.06, 0.39, 0.44, 0.46, 0.52, 3.16, 9.33])
#fids = np.nan_to_num(fids, 0.0)
accs = np.array([99.6, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
#accs = np.nan_to_num(accs, 0.0)

# plot
fig = plt.figure(figsize=(14.4, 4.8))

ax1 = plt.subplot(1,3,1)
ax1.plot(lambdas, psnrs, marker='p', color='slateblue')
ax1.set_xscale('log')
ax1.set_xlabel('$\\lambda$')
ax1.set_ylabel('PSNR')
ax1.grid(True, linestyle=':')

ax2 = plt.subplot(1,3,2)
ax2.plot(lambdas, fids, marker='D', color='crimson')
ax2.set_xscale('log')
ax2.set_xlabel('$\\lambda$')
ax2.set_ylabel('FID$*$')
ax2.grid(True, linestyle=':')

ax3 = plt.subplot(1,3,3)
ax3.plot(lambdas, accs, marker='h', color='darkorange')
ax3.set_xscale('log')
ax3.set_xlabel('$\\lambda$')
ax3.set_ylabel('Accuracy')
ax3.grid(True, linestyle=':')

plt.tight_layout()

# dump
if ag.save is not None:
    plt.savefig(ag.save)
else:
    plt.show()
