from scipy.interpolate import interp1d
import numpy as np
from numpy.random import rand
from progress import ProgressBar

class Sample(object):
    w = 1

class LinearSample(Sample):
    def __init__(self, xdat, ydat, w=1):
        self.xdat = xdat
        self.ydat = ydat
        self.f = interp1d(xdat,ydat,bounds_error=False,fill_value=0)
        self.w = w
    def __call__(self,x):
        return self.f(x)
    def __str__(self):
        return "w=%s, xdat=%s, ydat=%s" % (self.w, self.xdat, self.ydat)
    def __repr__(self):
        return "w=%s, xdat=%s, ydat=%s" % (self.w, self.xdat, self.ydat)


from numpy.random import choice

def trim_samples(samples,nsamp,pbar=False):

    weights = np.array([s.w for s in samples])
    weights /= max(weights)
    neff = np.sum(weights)
    n = len(weights)

    print "effective number of samples: " , neff, "/", n

    weights *= nsamp/neff

    if pbar: 
        progress_bar = ProgressBar(samples.size,message="trimming samples ")
    else:
        print "trimming samples"

    trimmed_samples = []
    for w, s in zip(weights,samples):
        if rand() < w:
            s.w = max(1.0,w)
            trimmed_samples.append(s)

        if pbar: 
            progress_bar()

    print "Samples trimmed from " , n, " to ", len(trimmed_samples)

    return trimmed_samples


