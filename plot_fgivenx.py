#!/usr/bin/python
from fgivenx.contours import load_contours
from matplotlib import pyplot as plt


# Parameters
# ----------

fig, axs = plt.subplots(2,5,sharex=True, sharey=True,figsize=(15,6))

for i, ax in enumerate(axs.flatten()):
    if i==10: 
        break
    contours = load_contours('contours/posterior' + str(i) + '.pkl')



    contours.plot(ax,colors=plt.cm.Greens_r)

    # Label axes
    label = 'Source ' + str(i+1)
    ax.text(0.9, 0.9, label,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes)


for ax in axs[-1,:]:
    ax.set_xlabel('$\log E$')

for ax in axs[:,0]:
    ax.set_ylabel('$\log \left[\\frac{dN}{dE}\\right]$')

# Remove spacing between subplots
#fig.subplots_adjust(wspace=0, hspace=0)
fig.tight_layout()


# Plot to file
# ------------
plt.savefig('plots/posterior.pdf', bbox_inches='tight', pad_inches=0.02, dpi=400)
