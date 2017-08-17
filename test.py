import fgivenx
import numpy
import matplotlib.pyplot

file_root = 'chains/test'
params = ['m1', 'c1']
samples, weights = fgivenx.samples_from_getdist_chains(file_root, params)

def f(x, theta):
    """ Simple y = m x + c function. """
    m, c = theta
    return m * x + c


xmin, xmax = -5, 5
nx = 100
x = numpy.linspace(xmin, xmax, nx)

x, y, z = fgivenx.compute_contours(f, x, samples, weights=weights)

fig, ax = matplotlib.pyplot.subplots()
cbar = fgivenx.plot(x, y, z, ax)

fig.savefig('plot.pdf')
