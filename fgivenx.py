#!/usr/bin/python

import fgivenx.planckStyle
import numpy as np
from matplotlib        import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl

def k2l(k):
    return 14000*k

def plot_single(x,P,f,ax,opts,pdf=True):

    x_limits = np.array([-4,-0.3])
    P_limits = np.array([2,4])
    if not pdf: fineness = 0.05
    else: fineness = 0.5
    vmax     = 3.5
    color    = plt.cm.Reds_r
    contour_levels = np.arange(0,4,fineness)

    # Initial processing
    # ------------------

    # Gaussian filter the mass
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(f, sigma=np.array(f.shape)/100.0 , order=0)

    # Convert to sigmas
    from scipy.special import erfinv
    f = np.sqrt(2)*erfinv(1-f)

    # compute k
    k = 10**x
    k_limits = 10**x_limits

    # compute ell
    l = k2l(k)
    l_limits = k2l(k_limits)


    # Plot contours
    # -------------

    # Plot the filled contours
    CS1 = ax.contourf(
            k,P,f,
            cmap=color,
            levels = contour_levels,vmin=0,vmax=vmax)

    if not pdf :
        ax.contourf(
                k,P,f,
                cmap=color,
                levels = contour_levels,vmin=0,vmax=vmax)

    # Plot some sigma-based contour lines
    CS2 = ax.contour(
            k,P,f, 
            colors='k', 
            linewidths=0.5,
            levels = [1,2,3],vmin=0,vmax=vmax)


    # Configure axes
    # --------------
    ax.set_ylim(P_limits)

    ax.set_xscale('log')
    ax.set_xlim(k_limits)

    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(l_limits)
    ax2.xaxis.set_major_formatter(ScalarFormatter())

    # define the P axis
    if 'l' in opts:
        ax.set_ylabel('$\log(10^{10}\\mathcal{P}_\\mathcal{R})$')
    else:
        for tic in ax.yaxis.get_major_ticks():
            tic.label1On = tic.label2On = False

    # define the lower k axis
    if 'b' in opts:
        ax.set_xlabel('$k/\\mathrm{Mpc}$')
    else:
        for tic in ax.xaxis.get_major_ticks():
            tic.label1On = tic.label2On = False

    # define the ell axis
    if 't' in opts:
        ax2.set_xlabel('$\\ell$')
    else:
        for tic in ax2.xaxis.get_major_ticks():
            tic.label1On = tic.label2On = False

    return [CS1,CS2]



pdf = True


fig,ax = plt.subplots(1,1)
fig.set_size_inches(3.5,3)

x = np.load('data/'+'combined'+'_x.npy')
P = np.load('data/'+'combined'+'_y.npy')
f = np.load('data/'+'combined'+'_z.npy')

CS = plot_single(x,P,f,ax,['t','b','l'],pdf)

cbaxis = fig.add_axes([0.9, 0.1, 0.03, 0.8]) 

colorbar = plt.colorbar(CS[0],cax=cbaxis,ticks = [0,1,2,3])
colorbar.ax.set_yticklabels(['$0\sigma$','$1\sigma$','$2\sigma$','$3\sigma$'])
colorbar.add_lines(CS[1])

fig.subplots_adjust(left=0.05,right=0.85)
plt.show()

output_root = "plots/single"

plt.savefig(output_root + ".pdf",bbox_inches='tight',pad_inches=0.02,dpi=400)


if not pdf:
    shell_command = "convert -density 400 " + output_root + ".pdf" + " -quality 100 " + output_root + ".png"
    print shell_command
    import os
    os.system(shell_command)



# Clear the figure
plt.clf()





filenames = np.array([
        ['0TT','1TT','2TT'],
        ['3TT','4TT','5TT'],
        ['6TT','7TT','8TT'],
        ])
options = np.array([
        [['l','t'],['t'],['t','r']],
        [['l'],    [],   ['r']    ],
        [['l','b'],['b'],['b','r']],
        ])



plt.rcParams['xtick.labelsize'] = 8 
plt.rcParams['ytick.labelsize'] = 8 

fig, axs = plt.subplots(3,3)

fig.set_size_inches(7.24,6)



for i in range(3):
    for j in range(3):
        fl = filenames[i,j]
        print "plotting "+fl
        x = np.load('data/'+fl+'_x.npy')
        P = np.load('data/'+fl+'_y.npy')
        f = np.load('data/'+fl+'_z.npy')
        CS = plot_single(x,P,f,axs[i,j],options[i,j],pdf)


plt.tight_layout()


cbaxis = fig.add_axes([0.9, 0.1, 0.03, 0.8]) 

colorbar = plt.colorbar(CS[0],cax=cbaxis,ticks = [0,1,2,3])
colorbar.ax.set_yticklabels(['$0\sigma$','$1\sigma$','$2\sigma$','$3\sigma$'])
colorbar.ax.tick_params(labelsize=12)
colorbar.add_lines(CS[1])

fig.subplots_adjust(left=0.05,right=0.85)



output_root = "array"

plt.savefig(output_root + ".pdf",bbox_inches='tight',pad_inches=0.02,dpi=400)


if not pdf:
    shell_command = "convert -density 400 " + output_root + ".pdf" + " -quality 100 " + output_root + ".png"
    print shell_command
    import os
    os.system(shell_command)


