""" This script computes the grid for contours of a function
    reconstruction plot.
    
    If one has
     * independent variable x
     * dependent variable y
     * functional form y = f(x,theta) parameterised by theta
    
    Assuming that you have obtained samples of theta from an MCMC
    process, we aim to compute:
    
                  /
    P( y | x ) =  | P( y = f(x,theta) | x, theta ) dtheta ,  (1)
                  /
    
    which gives our degree of knowledge for each y value given an x value.
    
    In fact, for a more representative plot, we are not actually
    interested in the value of the probability density (1), but in fact
    require the "iso-probablity posterior mass:"
    
                        /
    m( y | x ) =        | P(y'|x) dy'
                        /
                P(y'|x) < P(y|x)
    
    We thus need to compute this function on a rectangular grid of x and y's
    
    Any questions, please email Will Handley <wh260@mrao.cam.ac.uk>
"""

import pickle
import numpy
import matplotlib.pyplot

from progress import ProgressBar
from fgivenx.utils import PMF
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv

def load_contours(datafile):
    return pickle.load(open(datafile,'r'))

class Contours(object):
    def __init__(self, fsamples, x_range, nx=200, ny='nx', progress_bar = False):

        if ny == 'nx':
            ny = nx

        self.x = numpy.linspace(x_range[0], x_range[1], nx)
        slices  = compute_slices(fsamples,self.x,progress_bar)
        masses  = compute_masses(slices,progress_bar)
        self.y, self.z = compute_zs(ny,masses,progress_bar)

    def save(self,datafile):
        """ save class to file """
        pickle.dump(self,open(datafile, 'w'))

    def plot(self,ax,colors=matplotlib.pyplot.cm.Reds_r,smooth=False,contour_levels=None,fine_contour_level=None,fineness=0.5,linewidths=1.0):

        # define the default contour lines as 1,2
        if contour_levels == None:
            contour_levels = [1, 2]

        # Set up the fine contour gradation as 1 sigma above the levels above,
        # and with specified fineness
        if fine_contour_level == None:
            fine_contour_levels = numpy.arange(0, contour_levels[-1] + 1, fineness)

        # Create numpy arrays
        x = numpy.array(self.x)
        y = numpy.array(self.y)
        z = numpy.array(self.z)

        # Convert to sigmas
        z = numpy.sqrt(2) * erfinv(1 - z)

        # Gaussian filter if desired the sigmas by a factor of 1%
        if smooth:
            z = gaussian_filter(z, sigma=numpy.array(z.shape) / 100.0, order=0)

        # Plot the filled contours onto the axis ax
        for i in range(2):
            cbar = ax.contourf( x, y, z,
                cmap=colors, levels=fine_contour_levels)

        # Plot some sigma-based contour lines
        ax.contour( x, y, z,
            colors='k', linewidths=linewidths, levels=contour_levels)

        # Set limits on axes
        ax.set_xlim([min(x), max(x)])
        ax.set_ylim([min(y), max(y)])

        # Return the contours for use as a colourbar later
        return cbar


def compute_slices(fsamples,xs,pbar=False):
    """
    Convert a set of interpolation functions to a set of samples 
    from P( y(x) | x ) for several x's.

    Inputs:
      fsamples  : an array of functional samples
      xs        : an array of x coordinates
    Output:
     A 2D array containing samples from P
     """

    if pbar: 
        progress_bar = ProgressBar(len(xs),message="(1/3) computing slices ")
    else:
        print "(1/3) computing slices"

    slices = []
    for x in xs:
        slices.append([f(x) for f in fsamples])
        if pbar: 
            progress_bar()
                     
    return slices


def compute_masses(slices,pbar=False):

    if pbar: 
        progress_bar = ProgressBar(len(slices),message="(2/3) computing masses ")
    else: 
        print "(2/3) computing masses"
    masses = []

    for s in slices:
        masses.append( PMF(s) ) 
        if pbar: 
            progress_bar()

    return masses


def compute_zs(ny,masses,pbar=False):

    upper = max([m.upper for m in masses])
    lower = min([m.lower for m in masses])
    ys = numpy.linspace(lower,upper,ny)

    if pbar: 
        progress_bar = ProgressBar(len(ys),message="(3/3) computing zs     ")
    else: 
        print "(3/3) computing zs"

    zs = []
    for y in ys:
        zs.append([m(y) for m in masses])
        if pbar: 
            progress_bar()

    return ys, zs
