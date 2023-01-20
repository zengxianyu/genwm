import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family': 'monospace', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)

x = np.random.rand(10,10)
plt.pcolormesh(x, cmap='RdPu')
plt.colorbar()
#plt.show()
plt.savefig('colorbar.svg')
