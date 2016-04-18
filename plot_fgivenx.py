#!/usr/bin/python

import matplotlib.pyplot
from fgivenx.contours import load_contours


# Set up the grid of axes
fig, axes = matplotlib.pyplot.subplots(2,5,sharex=True, sharey=True,figsize=(16,6))

# plot the contours
for i, ax in enumerate(axes.flat):
    contours = load_contours('contours/posterior' + str(i) + '.pkl')
    colours = contours.plot(ax,
            colors=matplotlib.pyplot.cm.Greens_r,
            fineness = 0.25
            )

# Label axes
for i, ax in enumerate(axes.flat):
    label = 'Source ' + str(i+1)
    ax.text(0.9, 0.9, label,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes)

# x labels
for ax in axes[-1,:]:
    ax.set_xlabel('$\log E$')

# y labels
for ax in axes[:,0]:
    ax.set_ylabel('$\log \left[\\frac{dN}{dE}\\right]$')

# Tighten the axes together
fig.tight_layout()

# Add a colorbar (essential to do this after tight_layout)
cbar = fig.colorbar(colours, ax=axes.ravel().tolist(),ticks=[1,2,3],pad=0.01)
cbar.ax.set_yticklabels(['$1\sigma$', '$2\sigma$', '$3\sigma$'])

# Plot to file
matplotlib.pyplot.savefig('plots/posterior.pdf', bbox_inches='tight')
