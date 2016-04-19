#!/usr/bin/python

import matplotlib.pyplot
from fgivenx.contours import load_contours


# Set up the grid of axes
fig, ax = matplotlib.pyplot.subplots()

# plot the contours
contours = load_contours('contours/posterior.pkl')
colours = contours.plot(ax,
        colors=matplotlib.pyplot.cm.Greens_r,
        fineness = 0.25
        )

# x & y labels
ax.set_xlabel('$\log E$')
ax.set_ylabel('$\log \left[\\frac{dN}{dE}\\right]$')

# Add a colorbar (essential to do this after tight_layout)
cbar = fig.colorbar(colours, ax=ax, ticks=[1,2,3], pad=0.01)
cbar.ax.set_yticklabels(['$1\sigma$', '$2\sigma$', '$3\sigma$'])

# Plot to file
matplotlib.pyplot.savefig('plots/posterior.pdf', bbox_inches='tight')
