import pickle
import numpy
import matplotlib.pyplot

from progress import ProgressBar

import scipy.integrate

from fgivenx.utils import find_all_roots,PMF
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv

def load_contours(datafile):
    return pickle.load(open(datafile,'r'))

class Contours(object):
    def __init__(self, fsamples, x_range, nx, progress_bar = False):

        self.x = numpy.linspace(x_range[0], x_range[1], nx)
        slices  = compute_slices(fsamples,self.x,progress_bar)
        masses  = compute_masses(slices,progress_bar)
        self.y, self.z = compute_zs(self.x,masses,progress_bar)

    def save(self,datafile):
        """ save class to file """
        pickle.dump(self,open(datafile, 'w'))

    def plot(self,ax,colors=matplotlib.pyplot.cm.Reds_r):

        max_sigma = 3.5
        fineness = 0.1
        contour_levels = numpy.arange(0, 4, fineness)

        x = numpy.array(self.x)
        y = numpy.array(self.y)
        z = numpy.array(self.z)

        # Put the limits into an array
        x_limits = numpy.array([min(x), max(x)])
        y_limits = numpy.array([min(y), max(y)])

        # Gaussian filter the mass by a factor of 1%
        #z = gaussian_filter(z, sigma=numpy.array(z.shape) / 100.0, order=0)

        # Convert to sigmas
        z = numpy.sqrt(2) * erfinv(1 - z)


        # Plotting
        # --------
        # Plot the filled contours onto the axis ax
        print "Plotting filled contours"
        for i in range(2):
            CS1 = ax.contourf(
                x, y, z,
                cmap=colors,
                levels=contour_levels, vmin=0, vmax=max_sigma
                )

        # Plot some sigma-based contour lines
        print "Plotting contours"
        CS2 = ax.contour(
            x, y, z,
            colors='k',
            linewidths=1.0,
            levels=[1, 2], vmin=0, vmax=max_sigma
            )


        # Set limits on axes
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        # Colorbar
        #cbaxis = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        #
        #colorbar = plt.colorbar(CS1, ticks=[0, 1, 2, 3])
        #colorbar.ax.set_yticklabels(
        #    ['$0\sigma$', '$1\sigma$', '$2\sigma$', '$3\sigma$'])
        #colorbar.ax.tick_params(labelsize=18)
        #colorbar.add_lines(CS2)








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
        progress_bar = ProgressBar(len(fsamples),message="computing slices ")
    else:
        print "computing slices"

    slices = []
    for x in xs:
        slices.append([f(x) for f in fsamples])
        if pbar: 
            progress_bar()
                     
    return slices


def compute_masses(slices,pbar=False):

    if pbar: progress_bar = ProgressBar(len(slices),message="computing masses ")
    else: print "computing masses"
    masses = []

    for s in slices:
        masses.append( PMF(s) ) 
        if pbar: progress_bar()

    return masses


def compute_zs(xs,masses,pbar=False):
    upper = max([m.upper for m in masses])
    lower = min([m.lower for m in masses])
    n = len(xs)
    ys = numpy.linspace(lower,upper,n)
    zs = [[m(y) for m in masses] for y in ys]
    return ys,zs
