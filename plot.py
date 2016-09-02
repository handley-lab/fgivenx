""" Plot the contours.

    Note that you need to generate the contours first
    (see compute_contours.py for an example).
"""
import matplotlib.pyplot
from fgivenx.contours import Contours

# Set up the grid of axes
fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(1,1,1)

# plot the contours
contourfile = 'contours/posterior.pkl'
contours = Contours.load(contourfile)
colours = contours.plot(ax)

# x & y labels
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

# Add a colorbar (essential to do this after tight_layout)
cbar = fig.colorbar(colours, ax=ax, ticks=[1, 2, 3], pad=0.01)
cbar.ax.set_yticklabels(['$1\\sigma$', '$2\\sigma$', '$3\\sigma$'])

# Plot to file
matplotlib.pyplot.savefig('plots/posterior.pdf', bbox_inches='tight')
