import numpy
import matplotlib.pyplot
import fgivenx
import fgivenx.plot
from fgivenx.io import DKLCache

# Model definitions
# =================

# Define a simple straight line function, parameters theta=(m,c)
def f(x, theta):
    m, c = theta
    return m * x + c

numpy.random.seed(1)
# Posterior samples
nsamples = 1000
ms = numpy.random.normal(loc=-5,scale=1,size=nsamples)
cs = numpy.random.normal(loc=2,scale=1,size=nsamples) 
samples = numpy.array([(m,c) for m,c in zip(ms,cs)]).copy()

# Prior samples
ms = numpy.random.normal(loc=0,scale=5,size=nsamples)
cs = numpy.random.normal(loc=0,scale=5,size=nsamples) 
prior_samples = numpy.array([(m,c) for m,c in zip(ms,cs)]).copy()

# Computation
# ===========
# Examine the function over a range of x's
xmin, xmax = -2, 2
nx = 100
x = numpy.linspace(xmin, xmax, nx)


# Set the cache
cache = DKLCache('cache/test')
parallel = 'openmp'

# Compute the dkls
x, dkls = fgivenx.compute_kullback_liebler(f, x, samples, prior_samples, cache=cache, parallel=parallel, nprocs=8)

# Compute the contours
x, y, z = fgivenx.compute_contours(f, x, samples, cache=cache.posterior(), parallel=parallel, nprocs=8)
_, y_prior, z_prior = fgivenx.compute_contours(f, x, prior_samples, cache=cache.prior(), parallel=parallel, nprocs=8)


# Plotting
# ========
fig, axes = matplotlib.pyplot.subplots(2,2)

# Sample plot
ax = axes[0,1]
ax.plot(prior_samples.T[0],prior_samples.T[1],'b.')
ax.plot(samples.T[0],samples.T[1],'r.')
ax.set_xlabel('$m$')
ax.set_ylabel('$c$')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()

# Predictive posterior plot
ax = axes[0,0]
cbar = fgivenx.plot.plot(x, y_prior, z_prior, ax, colors=matplotlib.pyplot.cm.Blues_r,linewidths=0)
cbar = fgivenx.plot.plot(x, y, z, ax)
ax.set_xticklabels([])
ax.set_ylabel(r'$y = m x + c$')

# DKL plot
ax = axes[1,0]
ax.plot(x, dkls)
ax.set_ylim(bottom=0)
ax.set_xlabel('$x$')
ax.set_ylabel('$D_{KL}$')

ax.get_shared_x_axes().join(ax, axes[0,0])

fig.delaxes(axes[1,1])

fig.tight_layout()
fig.savefig('plot.pdf')
