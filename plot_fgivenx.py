#!/usr/bin/python
from fgivenx.plot_contours import plot_contours
from matplotlib import pyplot as plt


# Parameters
# ----------
root  = 'my_data'      # root name for files
xlabel = '$z$'
ylabel = '$w(z)$'

fig, ax = plt.subplots()
plot_contours(ax,root,colors=plt.cm.Greens_r)

# Label axes
ax.set_xlabel('$\log E$')
ax.set_ylabel('$\log \left[\\frac{dN}{dE}\\right]$')


# Plot to file
# ------------
output_root = "plots/" + root

# Save as pdf
print "Saving as pdf"
plt.savefig(output_root + ".pdf", bbox_inches='tight',
            pad_inches=0.02, dpi=400)
