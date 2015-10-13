#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

nx   = 200       # resolution in x direction
xmin = -np.pi    # minimum of x range
xmax =  np.pi    # maximum of x range 
x_limits = [xmin,xmax]

ny   = 200       # resolution in y direction 
ymin = -1.5      # minimum of y range        
ymax =  1.5      # maximum of y range        
y_limits = [ymin,ymax]

nresamp = 50     # resolution of kernel estimation 
                 # (shouldn't need to change this)


# Generate the samples
# --------------------
print "Generating samples"
from sample import randomSamples
nsamp   = 2000
samples = randomSamples(xmin,xmax,nsamp)


# Define the slices we're working on
# ----------------------------------
x = np.linspace(xmin,xmax,nx)
y = np.linspace(ymin,ymax,ny)


# Compute the data sets in each slice
# -----------------------------------
print "computing slices"
slices  = np.array([sample.f(x) for sample in samples]).T


# Compute the kernels
# -------------------
from kde import fast_kernel
#print "computing kernels"
#kernels = [ compute_kernel(s) for s in slices ]

print "computing fast kernels"
fast_kernels = [fast_kernel(s,np.linspace(ymin,ymax,nresamp)) for s in slices]

print "computing masses"
def compute_pmf(y,kernel):
    ny   = y.size
    pmf  = np.zeros(ny)

    prob = np.exp(kernel(y))

    ii = np.argsort(prob)
    cdf=0
    for i in ii:
        cdf+=prob[i]/ny
        pmf[i] = cdf

    return pmf/cdf


masses = np.array([ compute_pmf(y,kernel) for kernel in fast_kernels ])

from scipy.special import erfinv

masses = np.sqrt(2)*erfinv(1-masses.T)


# Plot
# ----
print "plotting"
import matplotlib.pyplot as plt


mask_level = 3.5

z = np.ma.array(masses, mask = masses>=mask_level )

color = plt.cm.Reds_r

fig, axs = plt.subplots(1,1)

cax = axs.contourf(x,y,z,30,cmap=color,vmin=0,vmax=3,rasterized=True)
axs.contour(x,y,z,30,cmap=color,vmin=0,vmax=3,rasterized=True)
cs = axs.contour(x,y,z, colors='k',levels = [1,2,3],vmin=0,vmax=3)

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


plt.savefig("temp.pdf",bbox_inches='tight',pad_inches=0.02,dpi=100)

plt.show()

import sys
sys.exit(0)

axs.contourf(x,y,probs.T)

plt.show()


import sys
sys.exit(0)


