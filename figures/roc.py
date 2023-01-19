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
ag.add_argument('--seen_ytrue', type=str, default=None, help='optional')
ag.add_argument('--seen_yscore', type=str, default=None, help='optional')
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
plt.plot([0,1], [0,1], 'k:', label='_nolegend_')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(ag.title)

legends = ['ours']
if ag.seen_ytrue and ag.seen_yscore:
	legends.append('seen')
	seen_ytrue = np.loadtxt(ag.seen_ytrue)
	seen_yscore = np.loadtxt(ag.seen_yscore)
	fpr, tpr, _ = roc_curve(seen_ytrue, seen_yscore, pos_label=1)
	plt.plot(fpr, tpr, marker='+')

if len(legends) > 1:
	plt.legend(legends)

plt.tight_layout()

# dump
if ag.save is not None:
	plt.savefig(ag.save)
else:
	plt.show()
