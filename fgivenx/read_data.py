import numpy as np
from sample import LinearSample,trim_samples
import csv
import sys


# Save the contours in a binary format
def save_contours(root,x,y,z):
    np.save('contours/'+root+'_x',x)
    np.save('contours/'+root+'_y',y)
    np.save('contours/'+root+'_z',z)

# Load the contours from a binary format
#
# This should be called as:
#    x,y,z = read_contours(<rootname>)
def read_contours(root):
    return [
            np.load('contours/'+root+'_x.npy'),
            np.load('contours/'+root+'_y.npy'),
            np.load('contours/'+root+'_z.npy')
            ]



def read_and_trim(filename,nsamp=0):

    # Read in all the samples
    # -----------------------
    print "Reading samples from file"
    samples = []; f = open(filename,'r')

    for line in f:
        line = line.split()                         # split the line into an array of strings 
        w    = float(line.pop(0))                   # pop the weight and add it to the array                 
        xy   = np.array([ float(c) for c in line ]) # extract the xy coordinates
        n    = xy.size/2                            # get the number of (x,y) coordines on this line
        x,y  = xy[:n],xy[n:]                        # extract the x and y coordinates from xy
        samples.append(LinearSample(x,y,w))         # create the sample and add to the array

    samples = trim_samples(np.array(samples))
    
    return samples
