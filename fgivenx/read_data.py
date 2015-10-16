import numpy as np
from sample import LinearSample
import csv

def read_and_trim(filename,nsamp):
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
        num+=1

    # renormalise
    weights = np.array(weights)/max(weights)
    # strip
    trimmed_samples = []
    for i in range(num):
        if np.random.rand()<weights[i] : trimmed_samples.append(samples[i])

    samples = np.array(trimmed_samples)
    print samples.size, " effective samples"

    if(samples.size<nsamp) :
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


