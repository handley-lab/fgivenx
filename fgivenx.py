#!/usr/bin/python

import numpy as np

nx   = 50
xmin = -np.pi
xmax =  np.pi

ny   = 50
ymin = -2.0
ymax =  2.0


# Generate the samples
# --------------------
print "Generating samples"
from sample import randomSamples
nsamp   = 20000
samples = randomSamples(xmin,xmax,nsamp)


# Define the slices we're working on
# ----------------------------------
x = np.linspace(xmin,xmax,nx)
y = np.linspace(ymin,ymax,ny)


# Compute the data sets in each slice
# -----------------------------------
print "computing slices"
slices  = np.array([sample.f(x) for sample in samples]).T


# Compute the kernels
# -------------------
from kde import compute_kernel,pdf
print "computing kernels"
kernels = [ compute_kernel(s) for s in slices ]



print "computing probabilities"
probs = np.array([ pdf(y,kernel) for kernel in kernels])


from scipy.optimize import brentq
def find_root_in_interval(array,i,j,kernel,p):
    if(i<0) : return y[0]
    elif(j>=array.size) : return y[array.size-1]
    else :
        return brentq(lambda x: pdf(x,kernel)-p, y[i],y[j])

print "computing masses"
def compute_pmf(y,kernel):

    ny = y.size
    prob = pdf(y,kernel)  # compute raw probabilities

    pmf = np.zeros(ny)

    # now we aim to construct a set of intervals where the probability distribution is greater than p
    k=0
    for p in prob:
        # find a lower bound
        i_l = 0
        while i_l<ny :

            while i_l<ny and prob[i_l]<p : i_l+=1
            l = find_root_in_interval(y,i_l-1,i_l,kernel,p)
            i_r = i_l

            # now find the next upper bound
            while i_r<ny and prob[i_r]>=p : i_r+=1
            r = find_root_in_interval(y,i_r-1,i_r,kernel,p)
            i_l=i_r

            pmf[k]+=kernel.integrate_box_1d(l,r)

        k+=1
        

    return pmf




masses = np.array([ compute_pmf(y,kernel) for kernel in kernels ])





import sys
sys.exit(0)




# Plot
# ----
print "plotting"
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1,1)

axs.contourf(x,y,masses.T)

plt.show()

axs.contourf(x,y,probs.T)

plt.show()


import sys
sys.exit(0)


