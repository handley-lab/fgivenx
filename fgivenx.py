from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.nonparametric.kde as KDETOOLS


# Definition of pi
pi = 4*np.arctan(1)

# Class definitions
# -----------------
class Sample(object):
    w = 1

class LinearSample(Sample):
    def __init__(self, xdat, ydat):
        self.f = interp1d(xdat,ydat)

# Compute a random sample
def randomSample():
    n = 11
    x = np.linspace(-pi,pi,n)
    for i in range(1,n-1):
        x[i]+=np.random.randn()*0.2
    y = np.random.choice([-1,1])*np.cos(-x) + np.random.randn(n)*0.2
    return LinearSample(x,y)

# Compute n samples from the above
def randomSamples(N=100):
    samples =[]
    for k in range(0,N,1):
        samples.append(randomSample())
    return np.array(samples)



# Compute a slice of the function at a valid x
def slice(samples,x):
    return np.array([sample.f(x) for sample in samples])

# Compute a kde estimator
#def kde(data,b=0.1):
#    return KernelDensity(bandwidth=b).fit(np.transpose([data]))

def compute_kernel(data):
    return KDETOOLS.KDEUnivariate(data).fit()

def pdf(x,kernel):
    return kernel.evaluate(x)





# Generate the samples
# --------------------
print "Generating samples"
nsamp=20000
samples = randomSamples(nsamp)
data = slice(samples,0)



#from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV




#print "computing kdes"
#grid = GridSearchCV(
#        KernelDensity(), 
#        {'bandwidth': np.logspace(-1, 1, 20)},
#        cv=20
#        )
#grid.fit(kde_data)

#b = grid.best_estimator_.bandwidth + 0.0
#print("best bandwidth: {0}".format(b))

xvals = np.linspace(pi/2,pi,20)

print "computing slices"
slices = np.array([sample.f(xvals) for sample in samples]).T
xnew = np.linspace(-1.5,1.5,1000)

for s in slices :
    print "computing kernel"
    kernel = KDETOOLS.KDEUnivariate(s)
    print "fitting kernel"
    kernel.fit()
    #ynew = kernel.evaluate(xnew)
    print "plotting"
    plt.plot(kernel.support,kernel.density)
    print "---------------------------"

plt.show()


import sys
sys.exit(0)

#ws = []
#for i in range(1,nsamp):
#    ws.append(1.0/nsamp)
#
#plt.hist(data,weights=ws,normed=True)


