#!/usr/bin/python
from fgivenx.read_data import read_contours

import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.special import erfinv

import cubehelix


from matplotlib import pyplot as plt
import os



print "Plotting contours"
print "-----------------"

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
print "Reading contours from file"
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
print "Plotting filled contours"
for i in range(2):
    CS1 = ax.contourf(
            x,y,z,
            cmap=color,
            levels = contour_levels,vmin=0,vmax=max_sigma)

# Plot some sigma-based contour lines
print "Plotting contours"
CS2 = ax.contour(
        x,y,z, 
        colors='k', 
        linewidths=1.0,
        levels = [1,2],vmin=0,vmax=max_sigma)


# Set limits on axes
ax.set_xlim(x_limits)
ax.set_ylim(y_limits)

# Label axes
ax.set_ylabel(ylabel, fontsize=18)
ax.set_xlabel(xlabel, fontsize=18)

# Colorbar
#cbaxis = fig.add_axes([0.9, 0.1, 0.03, 0.8]) 
#
colorbar = plt.colorbar(CS1,ticks = [0,1,2,3])
colorbar.ax.set_yticklabels(['$0\sigma$','$1\sigma$','$2\sigma$','$3\sigma$'])
colorbar.ax.tick_params(labelsize=18)
colorbar.add_lines(CS2)


#For the toy Model Data - want to add white functions to show performance:
# Need an xarray for the plots:
###xarray = np.linspace(0.0, 1.0,1000)
# yarray for the sin function: 
#(only need the below line and can then plot)
###yarray = [ np.sin(x*2*np.pi) for x in xarray ]
# yarray for the linear zig-zag:
#(create manually and finish by making yarray an np.array)
###yarray = []
###for x in xarray:
###    if x<=0.25:
###        yarray += [ 4.*x ]
###    elif x>0.25 and x<=0.75:
###        yarray += [ -4.*x +2. ]
###    else:
###        yarray += [ 4.*x -4. ]
###yarray = np.array(yarray)

#add the white plot
###ax.plot( xarray, yarray, color='white')


#fig.subplots_adjust(left=0.05,right=0.85)







# Plot to file
# ------------
output_root = "plots/" + root

# Save as pdf
print "Saving as pdf"
plt.savefig(output_root + ".pdf",bbox_inches='tight',pad_inches=0.02,dpi=400)

# Convert to png
print "Saving as png"
shell_command = "convert -density 400 " + output_root + ".pdf" + " -quality 100 " + output_root + ".png"
os.system(shell_command)

