from scipy.interpolate import interp1d
from numpy import array,linspace,cos
from numpy.random import randn as normal
from numpy.random import choice as choose

class Sample(object):
    w = 1

class LinearSample(Sample):
    def __init__(self, xdat, ydat):
        self.f = interp1d(xdat,ydat,bounds_error=False,fill_value=0)
    def __call__(self,x):
        return self.f(x)


# Compute a random sample
def randomSample(xmin,xmax):
    n = 11
    x = linspace(xmin,xmax,n)
    for i in range(1,n-1):
        x[i]+=normal()*0.2
    #y = cos(x+choose([0,1])) + normal(n)*0.2
    y = cos(x) + normal(n)*0.2
    return LinearSample(x,y)

# Compute n samples from the above
def randomSamples(xmin,xmax,N=100):
    samples =[]
    for k in range(0,N,1):
        samples.append(randomSample(xmin,xmax))
    return array(samples)

# Compute a slice of the function at a valid x
def slice(samples,x):
    return array([sample.f(x) for sample in samples])

