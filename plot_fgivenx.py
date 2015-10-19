#!/usr/bin/python
from fgivenx.read_data import read_contours

import os
import numpy as np

from matplotlib import pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.special import erfinv

import cubehelix



# Parameters
# ----------
root  = 'my_data'      # root name for files
color = cubehelix.cmap(reverse=False, start=0.0, rot=0.5, minLight=0.1)

xlabel = '$z$'
ylabel = '$w(z)$'

max_sigma     = 3.5

fineness = 0.1
contour_levels = np.arange(0,4,fineness)


# Read the data
# -------------
x,y,z = read_contours(root)


# Initial processing
# ------------------
# Put the limits into an array
x_limits = np.array([min(x),max(x)])
y_limits = np.array([min(y),max(y)])

# Gaussian filter the mass by a factor of 1%
z = gaussian_filter(z, sigma=np.array(z.shape)/100.0 , order=0)

# Convert to sigmas
z = np.sqrt(2)*erfinv(1-z)



# Plotting
# --------
# Initialise figure
fig,ax = plt.subplots(1,1)

# Plot the filled contours onto the axis ax
for i in range(2):
    CS1 = ax.contourf(
            x,y,z,
            cmap=color,
            levels = contour_levels,vmin=0,vmax=max_sigma)

# Plot some sigma-based contour lines
CS2 = ax.contour(
        x,y,z, 
        colors='k', 
        linewidths=0.5,
        levels = [1,2,3],vmin=0,vmax=max_sigma)


# Set limits on axes
ax.set_xlim(x_limits)
ax.set_ylim(y_limits)

# Label axes
ax.set_ylabel(ylabel)
ax.set_xlabel(xlabel)

# Colorbar
#cbaxis = fig.add_axes([0.9, 0.1, 0.03, 0.8]) 
#
colorbar = plt.colorbar(CS1,ticks = [0,1,2,3])
colorbar.ax.set_yticklabels(['$0\sigma$','$1\sigma$','$2\sigma$','$3\sigma$'])
colorbar.ax.tick_params(labelsize=12)
colorbar.add_lines(CS2)

#fig.subplots_adjust(left=0.05,right=0.85)







# Plot to file
# ------------
output_root = "plots/" + root

# Save as pdf
plt.savefig(output_root + ".pdf",bbox_inches='tight',pad_inches=0.02,dpi=400)

# Convert to png
shell_command = "convert -density 400 " + output_root + ".pdf" + " -quality 100 " + output_root + ".png"
print shell_command
os.system(shell_command)

