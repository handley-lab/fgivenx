import numpy as np
from progress import ProgressBar

# Compute the contour plot
def compute_contour_plot(samples,x,y):
    slices  = compute_slices(samples,x)
    weights = compute_weights(samples)
    kernels = compute_kernels(slices,weights)  
    masses  = compute_masses(kernels,y)
    return masses



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
    progress_bar = ProgressBar(fsamples.size,message="computing slices ")
    slices = []

    for f in fsamples:
        slices.append(f(xs))
        progress_bar()
                     
    return np.array(slices).T                   # return transpose

def compute_weights(fsamples):

    weights = np.array([f.w for f in fsamples])
    weights /= max(weights)
                     
    return weights


# compute_kernels
# ---------------
from weighted_kde import gaussian_kde
#
#   Converts samples from P( y(x) | x ) to kernel density estimates using weighted_kde.gaussian_kde
#
#   Inputs:
#     slices  : 2D array of samples from P( y(x) | x ) for several x's (as produced by compute_slices)
#   Output:
#     A 1D array of kernel density estimates of the distribution P( y(x) | x ) for each of the x's
#
def compute_kernels(slices,weights):
    progress_bar = ProgressBar(slices.size,message="computing kernels")
    kernels = []

    for s in slices:
        kernels.append(gaussian_kde(s,weights=weights))
        progress_bar()
                     
    return np.array(kernels)


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

    progress_bar = ProgressBar(kernels.size,message="computing masses ")
    masses = []

    for k in kernels:
        masses.append( compute_pmf(y,k) )         # compute M(x,y) for each value
        progress_bar()

    return np.array(masses).T             # return the transpose

# compute_sigmas
from scipy.special import erfinv
#
# Convert probability mass into sigma significance
def compute_sigmas(masses):
    return np.sqrt(2)*erfinv(1-masses)


