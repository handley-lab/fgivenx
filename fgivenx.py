#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from contours import compute_contour_plot

# Resolution settings
# -------------------
nx   = 200       # resolution in x direction
xmin = -np.pi    # minimum of x range
xmax =  np.pi    # maximum of x range 
x_limits = [xmin,xmax]

ny   = 200       # resolution in y direction 
ymin = -1.5      # minimum of y range        
ymax =  1.5      # maximum of y range        
y_limits = [ymin,ymax]



# Generate the samples
# --------------------
print "Generating samples"
from sample import randomSamples
nsamp   = 100 
samples = randomSamples(xmin,xmax,nsamp)

#import sys
#sys.exit(0)


# Compute the information for the contour plot
# ----------------------------------
x = np.linspace(xmin,xmax,nx)
y = np.linspace(ymin,ymax,ny)
z = compute_contour_plot(samples,x,y)



# Plot
# ----
print "plotting"


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
