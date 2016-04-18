import matplotlib.pyplot
import numpy
from fgivenx.read_data import read_contours

from scipy.ndimage import gaussian_filter
from scipy.special import erfinv

def plot_contours(ax,root,colors=matplotlib.pyplot.cm.Reds_r):
    print "Plotting contours"
    print "-----------------"

    max_sigma = 3.5
    fineness = 0.1
    contour_levels = numpy.arange(0, 4, fineness)


    # Read the data
    # -------------
    print "Reading contours from file"
    x, y, z = read_contours(root)


    # Initial processing
    # ------------------
    # Put the limits into an array
    x_limits = numpy.array([min(x), max(x)])
    y_limits = numpy.array([min(y), max(y)])

    # Gaussian filter the mass by a factor of 1%
    z = gaussian_filter(z, sigma=numpy.array(z.shape) / 100.0, order=0)

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
