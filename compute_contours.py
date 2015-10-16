#!/usr/bin/python

from numpy import linspace
from fgivenx.contours import compute_contour_plot
from fgivenx.read_data import read_and_trim,save_contours

# Settings
# --------
nx   = 100       # resolution in x direction
xmin = -4        # minimum of x range
xmax = -0.3      # maximum of x range 

ny   = 100       # resolution in y direction 
ymin = 2         # minimum of y range        
ymax = 4         # maximum of y range        

nsamp   = -1     # number of samples to keep ( <= 0 means keep all)

chains_file = 'chains/my_data.txt' # where the chains are kept
root        = 'my_data'            # the root name for the other files


# Computing contours
# ------------------

# Read chains file and convert into a set of interpolation functions
#   - We assume that the chains file has the format:
#      <weight>    <N x coordinates>   <N y coordinates>
#     in a space - separated file (just as in getdist)
#
#   - Note that you may need to use python or awk to process a raw chains file
#
#   - It is fine for each line to have different values of N, but there 
#     shouldn't be any additional irrelevant parameters
#   

samples = read_and_trim(chains_file)

# Compute a grid for making a contour plot
x = linspace(xmin,xmax,nx)
y = linspace(ymin,ymax,ny)
z = compute_contour_plot(samples,x,y)

# Save the contours to file for later use
save_contours(root,x,y,z)
