import numpy as np
from sample import LinearSample
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
    samples = []; weights = []; f = open(filename,'r')

    for line in f:
        line = line.split()                        # split the line into an array of strings 
        w    = float(line.pop(0))                   # pop the weight and add it to the array                 
        xy   = np.array([ float(c) for c in line ]) # extract the xy coordinates
        n    = xy.size/2                            # get the number of (x,y) coordines on this line
        x,y  = xy[:n],xy[n:]                        # extract the x and y coordinates from xy
        samples.append(LinearSample(x,y))          # create the sample and add to the array

    weights = np.array(weights)/max(weights)       # renormalise the weight

    
    # Now trim them into an equally weighted set of posteriors
    # --------------------------------------------------------
    print "Trimming samples"
    trimmed_samples = []
    for w,s in zip(weights,samples):
        if np.random.rand()<w : trimmed_samples.append(s)

    samples = np.array(trimmed_samples)

    print samples.size, " effective samples"

    if nsamp<=0 or samples.size<nsamp :
        return samples
    else:
        return np.random.choice(samples,nsamp,replace=False)
