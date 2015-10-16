#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from fgivenx.contours import compute_contour_plot
from fgivenx.read_data import read_and_trim,read_and_trim_combined

# Resolution settings
# -------------------
nx   = 100       # resolution in x direction
xmin = -4        # minimum of x range
xmax = -0.3      # maximum of x range 

ny   = 100       # resolution in y direction 
ymin = 2         # minimum of y range        
ymax = 4         # maximum of y range        

nsamp   = 100000000

x = np.linspace(xmin,xmax,nx)
y = np.linspace(ymin,ymax,ny)


filenames = np.array(['0TT','1TT','2TT','3TT','4TT','5TT','6TT','7TT','8TT'])
filenames = np.array([ 'chains/'+f+'_stripped.txt' for f in filenames])
evidences = np.array([-0.56741974E+004, -0.56745759E+004, -0.56749792E+004, -0.56759389E+004, -0.56763374E+004, -0.56770026E+004, -0.56768555E+004, -0.56776854E+004, -0.56776890E+004,])
evidences = np.exp(evidences-max(evidences))

samples = read_and_trim_combined(filenames,evidences,nsamp)
z = compute_contour_plot(samples,x,y)

np.save('data/'+'combined'+'_x',x)
np.save('data/'+'combined'+'_y',y)
np.save('data/'+'combined'+'_z',z)






for i in ['0TT','1TT','2TT','3TT','4TT','5TT','6TT','7TT','8TT'] :
#for i in ['8TT'] :
    # Read the samples
    # --------------------
    print "Reading samples from", i+'_stripped.txt'
    samples = read_and_trim('chains/'+i+'_stripped.txt',nsamp)

    # Compute the information for the contour plot
    # ----------------------------------
    z = compute_contour_plot(samples,x,y)

    # Print to file
    np.save('data/'+i+'_x',x)
    np.save('data/'+i+'_y',y)
    np.save('data/'+i+'_z',z)
