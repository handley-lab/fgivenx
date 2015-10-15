#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from contours import compute_contour_plot
from read_data import read_and_trim

# Resolution settings
# -------------------
nx   = 200       # resolution in x direction
xmin = -np.pi    # minimum of x range
xmax =  np.pi    # maximum of x range 

ny   = 200       # resolution in y direction 
ymin = -1.5      # minimum of y range        
ymax =  1.5      # maximum of y range        

nsamp   = 32000


# Read the samples
# --------------------
print "Reading samples"
samples = read_and_trim('data.dat',nsamp)

# Compute the information for the contour plot
# ----------------------------------
x = np.linspace(xmin,xmax,nx)
y = np.linspace(ymin,ymax,ny)
z = compute_contour_plot(samples,x,y)

# Print to file
np.save('data/x',x)
np.save('data/y',y)
np.save('data/z',z)
