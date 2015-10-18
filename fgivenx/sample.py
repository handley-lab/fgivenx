from scipy.interpolate import interp1d
import numpy as np
from numpy.random import rand
from progress import ProgressBar

class Sample(object):
    w = 1

class LinearSample(Sample):
    def __init__(self, xdat, ydat, w=1):
        self.f = interp1d(xdat,ydat,bounds_error=False,fill_value=0)
        self.w = w
    def __call__(self,x):
        return self.f(x)

from numpy.random import choice

def trim_samples(samples,nsamp,pbar=False):

    weights = np.array([s.w for s in samples])
    weights /= max(weights)
    neff = np.sum(weights)
    n    = weights.size 

    print "effective number of samples: " , neff, "/", n

    # Now trim off the ones that are too small
    ntarget = sum([ w if w<1.0/n else 1 for w in weights]) + 0.0

    if nsamp>0 and nsamp<ntarget:
        weights *= nsamp/neff
    else:
        weights *= n

    if pbar: progress_bar = ProgressBar(samples.size,message="trimming samples ")
    else: print "trimming samples"
    trimmed_samples = []
    for w,s in zip(weights,samples):
        if rand() < w:
            s.w = max(1.0,w)
            trimmed_samples.append(s)

        if pbar: progress_bar()

    trimmed_samples = np.array(trimmed_samples)
    print "Samples trimmed from " , n, " to ", trimmed_samples.size

    return trimmed_samples


