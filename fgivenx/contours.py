import numpy as np
from progress import ProgressBar

import scipy.integrate

from fgivenx.utils import find_all_roots, pairwise
from scipy.stats import gaussian_kde

def compute_contour_plot(fsamples,x,y,progress_bar=False):
    """From a set of function samples, compute the mass function on an x-y grid."""
    slices  = compute_slices(fsamples,x,progress_bar)
    kernels = compute_kernels(slices,progress_bar)  
    masses  = compute_masses(kernels,y,progress_bar)
    return masses



def compute_slices(fsamples,xs,pbar=False):
    """
    Convert a set of interpolation functions to a set of samples 
    from P( y(x) | x ) for several x's.

    Inputs:
      fsamples  : an array of interpolation functions (see sample.py)
      xs        : an array of x coordinates
    Output:
     A 2D array containing samples from P
     """

    if pbar: progress_bar = ProgressBar(len(fsamples),message="computing slices ")
    else: print "computing slices"
    slices = []

    for f in fsamples:
        slices.append([f(x) for x in xs])
        if pbar: progress_bar()
                     
    return np.array(slices).T                   # return transpose

def compute_weights(fsamples):

    weights = np.array([f.w for f in fsamples])
    weights /= max(weights)
                     
    return weights


# compute_kernels
# ---------------
#
#   Converts samples from P( y(x) | x ) to kernel density estimates using weighted_kde.gaussian_kde
#
#   Inputs:
#     slices  : 2D array of samples from P( y(x) | x ) for several x's (as produced by compute_slices)
#   Output:
#     A 1D array of kernel density estimates of the distribution P( y(x) | x ) for each of the x's
#
def compute_kernels(slices,pbar=False):
    if pbar: progress_bar = ProgressBar(slices.size,message="computing kernels")
    else: print "computing kernels"
    kernels = []

    for s in slices:
        kernels.append(gaussian_kde(s))
        if pbar: progress_bar()
                     
    return np.array(kernels)


# compute_pmf
# -----------
#   Computes the 'probability mass function' of a probability density
#   function
#
#          /
#   M(p) = |          P(y) dy
#          / P(y) < p
#
#   This is the cumulative distribution function expressed as a
#   function of the probability
#
#   We actually aim to compute M(y), which indicates the amount of
#   probability contained outside the iso-probability contour passing
#   through y
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
    pmf = []

    d = ys[0] - ys[-1]
    yl = ys[0] - d*100
    yu = ys[-1] + d*100

    # compute the probablities
    ps = kernel(ys)
    p_grid = [kernel(yl)] + list(ps) + [kernel(yu)]
    y_grid = [yl] + list(ys) + [yu]

    for p in ps:
        y0s = find_all_roots(y_grid,p_grid,p,kernel)
        if len(y0s) % 2 == 0:
            lowers = [-np.inf] + y0s[::2]
            uppers = y0s[1::2] + [-np.inf]

            total = 0.0
            for l, u in zip(lowers,uppers):
                total += kernel.integrate_box_1d(l,u)
            pmf.append(1-total)
        elif len(y0s)==1:
            pmf.append(1.0)
        else:
            pmf.append(1.0)
            
    return pmf

#    nys  = ys.size          # get the number of y values
#    pmf  = np.zeros(nys)    # initialise the M(y) with zeros
#    prob = kernel(ys)       # Compute the probablities at each point
#    ii   = np.argsort(prob) # Compute sorted indices
#    cdf  = 0                # Initialise cumulative density functions
#
#    for i in ii:            # for each y value compute the cdf as a function of p
#        cdf+=prob[i]/nys
#        pmf[i] = cdf
#
#    return pmf/cdf          # return it as normalised



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
def compute_masses(kernels,y,pbar=False):

    if pbar: progress_bar = ProgressBar(kernels.size,message="computing masses ")
    else: print "computing masses"
    masses = []

    for k in kernels:
        masses.append( compute_pmf(y,k) )         # compute M(x,y) for each value
        if pbar: progress_bar()

    return np.array(masses).T             # return the transpose
