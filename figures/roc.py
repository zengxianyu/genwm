import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import rich
console = rich.get_console()

# config
font = {'family': 'monospace', 'weight': 'normal', 'size': 20}
matplotlib.rc('font', **font)

# argparse
ag = argparse.ArgumentParser()
ag.add_argument('--ytrue', type=str, required=True,
                help='y_true data file for roc_curve')
ag.add_argument('--yscore', type=str, required=True,
                help='y_score data file for roc_curve')
ag.add_argument('--color', type=str, default='crimson')
ag.add_argument('--title', type=str, default='')
ag.add_argument('--save', type=str, default=None)
ag = ag.parse_args()
console.print(ag)

# read data
y_true = np.loadtxt(ag.ytrue)
print('y_true.shape', y_true.shape)
y_score = np.loadtxt(ag.yscore)
print('y_score.shape', y_score.shape)

# calculate
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
console.print('FPR', fpr)
console.print('TPR', tpr)

# plot
fig = plt.figure(figsize=(5.4, 4.8))
plt.plot(fpr, tpr, marker='p', color=ag.color)
plt.plot([0,1], [0,1], 'k:')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(ag.title)

plt.tight_layout()

# dump
if ag.save is not None:
	plt.savefig(ag.save)
else:
	plt.show()
