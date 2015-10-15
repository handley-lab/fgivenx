#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

x = np.load('data/x.npy')
y = np.load('data/y.npy')
z = np.load('data/z.npy')

x_limits = [min(x),max(x)]
y_limits = [min(y),max(y)]


# Plot
# ----
print "Plotting"


mask_level     = 3.5
fineness       = 0.5#3.0/10.0
contour_levels = np.arange(0,4,fineness)


z = np.ma.array(z, mask = z>=mask_level )

color = plt.cm.Reds_r

fig, axs = plt.subplots(1,1)

if fineness < 0.5 : axs.contour(x,y,z,cmap=color,levels = contour_levels,vmin=0,vmax=3)
cax = axs.contourf(x,y,z,cmap=color,levels = contour_levels,vmin=0,vmax=3)

cs = axs.contour(x,y,z, colors='k', levels = [1,2,3],vmin=0,vmax=3)

#for c in cax.collections:
#    c.set_rasterized(True)

cbar = fig.colorbar(cax, ticks = [0,1,2,3])
cbar.ax.set_yticklabels(['$0\sigma$','$1\sigma$','$2\sigma$','$3\sigma$'])

# define the y axis
axs.set_ylabel('$\log(10^{10}\\mathcal{P}_\\mathcal{R})$')
axs.set_ylim(y_limits)

# define the lower x axis
axs.set_xlabel('$k/\\mathrm{Mpc}$')
#axs.set_xscale('log') # set log scale 
axs.set_xlim(x_limits)


def k2l(k):
    return 14000*k


# define the upper x axis
axs2 = axs.twiny()
axs2.set_xlabel('$\\ell$')
#axs2.set_xscale('log')
from matplotlib.ticker import ScalarFormatter
x_1,x_2 = axs.get_xlim()
axs2.xaxis.set_major_formatter(ScalarFormatter())
axs2.set_xlim((k2l(x_1),k2l(x_2)))
axs2.set_ylim(y_limits)


plt.savefig("temp.pdf",bbox_inches='tight',pad_inches=0.02,dpi=400)

plt.show()
