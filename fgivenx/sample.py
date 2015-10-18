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

def trim_samples(samples):

    weights = np.array([s.w for s in samples])
    weights /= max(weights)
    neff = np.sum(weights)
    n    = weights.size 

    print "effective number of samples: " , neff, "/", n

    weights*=np.sum(weights)

    progress_bar = ProgressBar(samples.size,message="trimming samples ")
    trimmed_samples = []
    for w,s in zip(weights,samples):
        if rand() < w:
            s.w = max(1.0,w)
            trimmed_samples.append(s)
        progress_bar()


    weights = np.array([s.w for s in trimmed_samples])
    weights /= max(weights)


    print "Samples trimmed from " , n, " to ", weights.size

    return np.array(trimmed_samples)
