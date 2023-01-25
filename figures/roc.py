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
ag.add_argument('--unseen_ytrue', type=str, default=None, help='optional')
ag.add_argument('--unseen_yscore', type=str, default=None, help='optional')
ag.add_argument('--wang_ytrue', type=str, default=None)
ag.add_argument('--wang_yscore', type=str, default=None)
ag.add_argument('--grag_ytrue', type=str, default=None)
ag.add_argument('--grag_yscore', type=str, default=None)
ag = ag.parse_args()
console.print(ag)

# read data
y_true = np.loadtxt(ag.ytrue)
print('y_true.shape', y_true.shape)
y_score = np.loadtxt(ag.yscore)
print('y_score.shape', y_score.shape)

# plot
fig = plt.figure(figsize=(5.4, 4.8))
legends = []

if ag.seen_ytrue and ag.seen_yscore:
	legends.append('seen')
	seen_ytrue = np.loadtxt(ag.seen_ytrue)
	seen_yscore = np.loadtxt(ag.seen_yscore)
	fpr, tpr, _ = roc_curve(seen_ytrue, seen_yscore, pos_label=1)
	plt.plot(fpr, tpr, marker='x')

if ag.unseen_ytrue and ag.unseen_yscore:
    legends.append('unseen')
    unseen_ytrue = np.loadtxt(ag.unseen_ytrue)
    unseen_yscore = np.loadtxt(ag.unseen_yscore)
    fpr, tpr, _ = roc_curve(unseen_ytrue, unseen_yscore, pos_label=1)
    plt.plot(fpr, tpr, marker=None)

if ag.wang_ytrue and ag.wang_yscore:
    legends.append('Wang et al.')
    wang_ytrue = np.loadtxt(ag.wang_ytrue)
    wang_yscore = np.loadtxt(ag.wang_yscore)
    fpr, tpr, _ = roc_curve(wang_ytrue, wang_yscore, pos_label=1)
    plt.plot(fpr, tpr, marker=None)

if ag.grag_ytrue and ag.grag_yscore:
    legends.append('Gragnaniello et al.')
    grag_ytrue = np.loadtxt(ag.grag_ytrue)
    grag_yscore = np.loadtxt(ag.grag_yscore)
    fpr, tpr, _ = roc_curve(grag_ytrue, grag_yscore, pos_label=1)
    plt.plot(fpr, tpr, marker=None)

if ag.ytrue and ag.yscore:
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    console.print('FPR', fpr)
    console.print('TPR', tpr)
    plt.plot(fpr, tpr, marker='p', color=ag.color, linewidth=3)
    plt.plot([0,1], [0,1], 'k:', label='_nolegend_')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(ag.title)
    legends.append('ours')


plt.legend(legends, fontsize=12)
plt.tight_layout()

# dump
if ag.save is not None:
	plt.savefig(ag.save)
else:
	plt.show()
