""" Plot the contours.

    Note that you need to generate the contours first
    (see examples/group_compute_contours.py for an example).
"""
import matplotlib.pyplot
from fgivenx.contours import Contours

# Set up the grid of axes
fig, axes = matplotlib.pyplot.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))

# plot the contours
for i, ax in enumerate(axes.flat):
    contourfile = 'contours/posterior' + str(i) + '.pkl'
    contours = Contours.load(contourfile)
    colours = contours.plot(ax)

# x labels
for ax in axes[-1, :]:
    ax.set_xlabel('$x$')

# y labels
for ax in axes[:, 0]:
    ax.set_ylabel('$y$')

# Tighten the axes together
fig.tight_layout()

# Add a colorbar (essential to do this after tight_layout)
cbar = fig.colorbar(colours, ax=axes.ravel().tolist(), ticks=[1, 2, 3], pad=0.01)
cbar.ax.set_yticklabels(['$1\\sigma$', '$2\\sigma$', '$3\\sigma$'])

# Plot to file
matplotlib.pyplot.savefig('plots/posterior_group.pdf', bbox_inches='tight')
