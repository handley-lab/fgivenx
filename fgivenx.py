from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


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
    y = np.cos(-x) + np.random.randn(n)*0.2
    return LinearSample(x,y)

# Compute n samples from the above
def randomSamples(N=100):
    samples =[]
    for k in range(1,N,1):
        samples.append(randomSample())
    return np.array(samples)



# Compute a slice of the function at a valid x
def slice(samples,x):
    slice_list = list()
    for sample in samples:
        val = sample.f(x)
        slice_list.append(val)
    return np.array(slice_list)

# Compute a kde estimator
def kde(data,b=0.1):
    return KernelDensity(bandwidth=b).fit(np.transpose([data]))





# Generate the samples
# --------------------
print "Generating samples"
nsamp=5000
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



for x in [0,0.3,0.6,0.9,1.2]:
    print x
    data = slice(samples,x)
    kernel = kde(data,0.1)
    xnew = np.linspace(-1.5,1.5,1000)
    ynew = np.exp(kernel.score_samples(np.transpose([xnew])))
    plt.plot(xnew,ynew)

plt.show()


import sys
sys.exit(0)

#ws = []
#for i in range(1,nsamp):
#    ws.append(1.0/nsamp)
#
#plt.hist(data,weights=ws,normed=True)


def compute_kde(samples,x):
    return gaussian_kde(slice(samples,x)) 






# Plot the samples
# ----------------
print "Plotting histograms"

xnew = np.linspace(0.5,1.5,1000)
for xi in [0.0,0.1,0.2,0.3]:
    print xi
    kernel = compute_kde(samples,xi)
    print "plotting"
    plt.plot(xnew,kernel(xnew), '-')


plt.show()
import sys
sys.exit(0)

plt.hist(slice(samples,0))
plt.hist(slice(samples,1))
plt.hist(slice(samples,2))
plt.hist(slice(samples,3))
plt.show()

plt.clf() # clear the figure

xnew = np.linspace(-pi,pi,100)
plt.plot(xnew,np.cos(xnew), '-')

for sample in samples:
    plt.plot(xnew,sample.f(xnew), '-')

plt.show()



