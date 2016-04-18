#!/usr/bin/python
from fgivenx.contours import load_contours
from matplotlib import pyplot as plt


# Parameters
# ----------
contours = load_contours('contours/posterior1.pkl')

fig, ax = plt.subplots()
contours.plot(ax,colors=plt.cm.Greens_r)

# Label axes
ax.set_xlabel('$\log E$')
ax.set_ylabel('$\log \left[\\frac{dN}{dE}\\right]$')


# Plot to file
# ------------
plt.savefig("plots/posterior1.pdf", bbox_inches='tight', pad_inches=0.02, dpi=400)
