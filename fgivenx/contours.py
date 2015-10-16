import numpy as np
from progress import update_progress

# Compute the contour plot
def compute_contour_plot(samples,x,y):
    slices  = compute_slices(samples,x)
    kernels = compute_kernels(slices)  
    masses  = compute_masses(kernels,y)
    return masses
    #sigmas  = compute_sigmas(masses)   
    #return sigmas



# compute_slices
# --------------
#   Converts a set of interpolation functions to a set of samples 
#   from P( y(x) | x ) for several x's
#
#   Inputs:
#     fsamples  : an array of interpolation functions (see sample.py)
#     xs        : an array of x coordinates
#   Output:
#     A 2D array containing samples from P
#
def compute_slices(fsamples,xs):

    nxs = xs.size                      # get the size of x array
    nsamp = fsamples.size              # get the number of samples
    slices = np.empty([nsamp,nxs])     # initialise empty array to return

    for i in range(nsamp):                                   
        update_progress((i+1.0)/nsamp, "computing slices")   # progress bar
        slices[i] = fsamples[i](xs)                          # compute slice

    return slices.T                   # return transpose



# compute_kernels
# ---------------
from scipy.stats import gaussian_kde
#
#   Converts samples from P( y(x) | x ) to kernel density estimates using scipy.stats.gaussian_kde
#
#   Inputs:
#     slices  : 2D array of samples from P( y(x) | x ) for several x's (as produced by compute_slices)
#   Output:
#     A 1D array of kernel density estimates of the distribution P( y(x) | x ) for each of the x's
#
def compute_kernels(slices):
    print "computing kernels"
    return np.array([ gaussian_kde(s) for s in slices ])


# compute_pmf
# -----------
#   Computes the 'probability mass function' of a probability density function
#
#          /
#   M(p) = |          P(y) dy
#          / P(y) < p
#
#   This is the cumulative distribution function expressed as a function of the probability
#
#   We actually aim to compute M(y), which indicates the amount of probability contained outside the iso-probability contour at y
#
#
#  ^ P(y)
#  |                       .....
#  |                     .       .
# p|- - - - - - - - - - . - - - - . - - - - - - - - - - - 
#  |                   .#         #.
#  |                  .##         ##.
#  |                  .##         ##.
#  |                 .###         ###.       M(p) is the shaded area
#  |                 .###         ###.
#  |                 .###         ###.
#  |                .####         ####.
#  |                .####         ####.
#  |              ..#####         #####..
#  |          ....#######         #######....
#  |         .###########         ###########.
#  +-----------------------------------------------------> y 
#
#   ^ M(p)                        ^ M(y)                   
#   |                             |                        
#  1|                +++         1|         + 
#   |               +             |        + +
#   |       ++++++++              |       +   +            
#   |     ++                      |     ++     ++          
#   |   ++                        |   ++         ++        
#   |+++                          |+++             +++     
#   +---------------------> p     +---------------------> y
#  0                   1                                   
#
#   Inputs:
#     kernel : a kernel density estimate of P(y)
#     ys     : set of y values
#   Output:
#     1D array of M(y) values at each ys
#
def compute_pmf(ys,kernel):
    nys   = ys.size         # get the number of y values
    pmf  = np.zeros(nys)    # initialise the M(y) with zeros
    prob = kernel(ys)       # Compute the probablities at each point
    ii   = np.argsort(prob) # Compute sorted indices
    cdf  = 0                # Initialise cumulative density functions

    for i in ii:            # for each y value compute the cdf as a function of p
        cdf+=prob[i]/nys
        pmf[i] = cdf
    return pmf/cdf          # return it as normalised



# compute_masses
# --------------
#   Converts a set of functions P( y(x) | x ) into a grid of probability mass functions
#
#   Inputs:
#     kernels   : array of kernel density estimates of the distribution P( y(x) | x ) for each of the x's
#                 (as produced by compute_kernels)
#     ys        : an array of y coordinates
#   Output:
#     A 2D array indicating M(x,y) where for each x, M(y) is the probability mass function of P( y(x) | x )
#
def compute_masses(kernels,y):

    nx = kernels.size           # get number of x coords
    ny = y.size                 # get number of y coords
    masses = np.empty([nx,ny])  # initialise masses at zero

    for i in range(nx):
        update_progress((i+1.0)/nx, "computing masses") # progress bar
        masses[i] = compute_pmf(y,kernels[i])           # compute M(x,y) for each value
    return masses.T             # return the transpose

# compute_sigmas
from scipy.special import erfinv
#
# Convert probability mass into sigma significance
def compute_sigmas(masses):
    return np.sqrt(2)*erfinv(1-masses)


