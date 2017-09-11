import numpy
import matplotlib.pyplot
import fgivenx
import fgivenx.plot

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
cache = 'cache/test'

# Compute the dkls
x, dkls = fgivenx.compute_kullback_liebler(f, x, samples, prior_samples, cache=cache)

# Compute the contours
x, y, z = fgivenx.compute_contours(f, x, samples, cache=cache)
_, y_prior, z_prior = fgivenx.compute_contours(f, x, prior_samples, cache=cache+'_prior')

x, fsamps = fgivenx.compute_samples(f, x, samples, cache=cache)
x, prior_fsamps = fgivenx.compute_samples(f, x, prior_samples, cache=cache+'_prior')

# Plotting
# ========
fig, axes = matplotlib.pyplot.subplots(2,2)

# Sample plot
# -----------
ax = axes[0,0]
ax.set_ylabel(r'$c$')
ax.set_xlabel(r'$m$')
ax.plot(prior_samples.T[0],prior_samples.T[1],'b.')
ax.plot(samples.T[0],samples.T[1],'r.')

# Line plot
# ---------
ax = axes[0,1]
ax.set_ylabel(r'$y = m x + c$')
fgivenx.plot.plot_lines(x, prior_fsamps, ax, color='b')
fgivenx.plot.plot_lines(x, fsamps, ax, color='r')
ax.set_xticklabels([])

# Predictive posterior plot
# -------------------------
ax = axes[1,1]
ax.set_ylabel(r'$P(y|x)$')
ax.set_xlabel(r'$x$')
cbar = fgivenx.plot.plot(x, y_prior, z_prior, ax, colors=matplotlib.pyplot.cm.Blues_r,linewidths=0)
cbar = fgivenx.plot.plot(x, y, z, ax)

# DKL plot
# --------
ax = axes[1,0]
ax.set_ylabel(r'$D_\mathrm{KL}$')
ax.set_xlabel(r'$x$')
ax.plot(x, dkls)
ax.set_ylim(bottom=0)

axes[0,0].get_shared_x_axes().join(axes[0,0], axes[1,0], axes[1,1])

fig.tight_layout()
fig.savefig('plot.png')
