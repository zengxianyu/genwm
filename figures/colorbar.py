import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family': 'monospace', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)

vmax = [15.588457268119896,
    33.075670817082454,
    49.90991885387112,
    36.373066958946424]
for v in vmax:
    x = np.random.rand(10,10) * v
    plt.pcolormesh(x, cmap='RdPu')
    plt.colorbar()
    #plt.show()
    plt.savefig(f'colorbar-{v}.svg')
