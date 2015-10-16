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
            np.load('contours/'+root+'_x'),
            np.load('contours/'+root+'_y'),
            np.load('contours/'+root+'_z')
            ]



def read_and_trim(filename,nsamp=0):

    # Read in all the samples
    # -----------------------
    print "Reading samples from file"
    samples = []; weights = []; f = open(filename,'r')

    for line in f:
        line = line.split()                        # split the line into an array of strings 
        weights.append(float(line.pop(0)))         # pop the weight and add it to the array                 
        xy  = np.array([ float(c) for c in line ]) # extract the xy coordinates
        n   = xy.size/2                            # get the number of (x,y) coordines on this line
        x,y = xy[:n],xy[n:]                        # extract the x and y coordinates from xy
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








def read_and_trim_combined(filenames,evidences,nsamp):
    samples_array =[]
    weights_array =[]
    for filename in filenames:
        print "reading "+filename
        samples = []
        weights = []
        num=0
        f = open(filename,'r')
        for line in f:
            line = line.split()
            w = float(line.pop(0))
            xy = np.array([ float(c) for c in line ])
            n  = xy.size/2
            x  = xy[:n]
            y  = xy[n:]
            weights.append(w)
            samples.append(LinearSample(x,y))

        weights_array.append(np.array(weights))
        samples_array.append(np.array(samples))

    print "adjusting weights"
    for i in range(evidences.size): 
        weights_array[i]*=evidences[i]

    # renormalise
    maxw = max([ max(w) for w in weights_array])
    for i in range(evidences.size):
        weights_array[i] = weights_array[i]/maxw

    # strip
    print "stripping"
    trimmed_samples = []
    for weights,samples in zip(weights_array,samples_array):
        for w,s in zip(weights,samples):
            if np.random.rand()<w : trimmed_samples.append(s)

    samples = np.array(trimmed_samples)
    print samples.size, " effective samples"

    if(samples.size<nsamp) :
        return samples
    else:
        return np.random.choice(samples,nsamp,replace=False)


