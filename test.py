import fgivenx
import fgivenx.plot
import numpy
import matplotlib.pyplot


# Define a simple straight line function, parameters theta=(m,c)
def f(x, theta):
    m, c = theta
    return m * x + c

# Create some sample gradient and intercepts
nsamples = 1000
ms = numpy.random.normal(loc=1,size=nsamples)
cs = numpy.random.normal(loc=0,size=nsamples) 
samples = numpy.array([(m,c) for m,c in zip(ms,cs)])

# Examine the function over a range of x's
xmin, xmax = -2, 2
nx = 100
x = numpy.linspace(xmin, xmax, nx)

# Compute the contours
x, y, z = fgivenx.compute_contours(f, x, samples, parallel='mpi')

# Plot 
fig, ax = matplotlib.pyplot.subplots()
cbar = fgivenx.plot.plot(x, y, z, ax)

fig.savefig('plot.pdf')
