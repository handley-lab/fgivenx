import scipy
import numpy
import matplotlib.pyplot


def plot(x, y, z, ax, **kwargs):
    """ Plot computed contours.

        Parameters
        ----------
        x, y, z : numpy arrays
            See arguments to matplotlib.pyplot.contour
        ax: matplotlib.axes._subplots.AxesSubplot
            Axes to plot the contours onto.
            Typically generated with:
                fig, ax = matplotlib.pyplot.subplots()

        Keywords
        --------
        colors: matplotlib.colors.LinearSegmentedColormap
            (Default: matplotlib.pyplot.cm.Reds_r)
            Color scheme to plot with. Recommend plotting in reverse
        smooth: bool
            (Default: False)
            Whether to smooth the contours.
        contour_line_levels: List[float]
            (Default: [1,2])
            Contour lines to be plotted.
        linewidths: float
            (Default: 0.3)
            Thickness of contour lines
        contour_color_levels: List[float]
            (Default: numpy.arange(0, contour_line_levels[-1] + 1, fineness))
            Contour color levels.
        fineness: float
            (Default: 0.1)
            Spacing of contour color levels.
        lines: bool
            (Default: True)


        Returns
        -------
        cbar: matplotlib.contour.QuadContourSet
            Colors to create a global colour bar

        Functionality mostly determined by modifications to ax
    """
    # Get inputs
    colors = kwargs.pop('colors', matplotlib.pyplot.cm.Reds_r)
    smooth = kwargs.pop('smooth', False)

    linewidths = kwargs.pop('linewidths', 0.3)
    contour_line_levels = kwargs.pop('contour_line_levels', [1, 2, 3])

    fineness = kwargs.pop('fineness', 0.5)
    default_color_levels = numpy.arange(0, contour_line_levels[-1] + 1,
                                        fineness)
    contour_color_levels = kwargs.pop('contour_color_levels',
                                      default_color_levels)

    alpha = kwargs.pop('alpha', 1.0)
    lines = kwargs.pop('lines', True)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    # Convert to sigmas
    z = numpy.sqrt(2) * scipy.special.erfinv(1 - z)

    # Gaussian filter if desired the sigmas by a factor of 1%
    if smooth:
        z = scipy.ndimage.gaussian_filter(z, sigma=numpy.array(z.shape)/100.0,
                                          order=0)

    # Plot the filled contours onto the axis ax
    cbar = ax.contourf(x, y, z, cmap=colors, levels=contour_color_levels, alpha=alpha)

    # Remove those annoying white lines
    for c in cbar.collections:
        c.set_edgecolor("face")

    # Plot some sigma-based contour lines
    if lines:
        ax.contour(x, y, z, colors='k', linewidths=linewidths,
                   levels=contour_line_levels)

    # Return the contours for use as a colourbar later
    return cbar

def plot_lines(x, fsamps, ax, downsample=100, **kwargs):
    indices = numpy.random.choice(len(fsamps.T), downsample, replace=False)
    linewidth = kwargs.pop('linewidth',0.1)
    color = kwargs.pop('color','k')
    for y in fsamps.T[indices]:
        ax.plot(x, y, linewidth=linewidth, color=color, **kwargs)
