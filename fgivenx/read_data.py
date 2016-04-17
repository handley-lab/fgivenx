import numpy as np
from sample import LinearSample,trim_samples
from progress import ProgressBar


# Save the contours in a binary format
def save_contours(root,x,y,z):
    print "saving contours to files contours/" + root + "_{x,y,z}.npy"
    np.save('contours/'+root+'_x',x)
    np.save('contours/'+root+'_y',y)
    np.save('contours/'+root+'_z',z)

# Load the contours from a binary format
#
# This should be called as:
#    x,y,z = read_contours(<rootname>)
def read_contours(root):
    print "reading contours from files contours/" + root + "_{x,y,z}.npy"
    return [
            np.load('contours/'+root+'_x.npy'),
            np.load('contours/'+root+'_y.npy'),
            np.load('contours/'+root+'_z.npy')
            ]






def get_samples(filename, function, chosen_parameters, nsamp=0,pbar=False):

    # Read in all the samples
    # -----------------------
    num_lines = sum(1 for line in open(filename))
    if pbar: progress_bar = ProgressBar(num_lines,message="reading samples  ")
    else: print "reading samples"
    samples = []; f = open(filename,'r')

    for line in f:
        line   = line.split()                         # split the line into an array of strings 
        w      = float(line.pop(0))                   # pop the weight and add it to the array                 
        logL   = float(line.pop(0))/-2                # pop the loglikelihood
        params = np.array([ float(c) for c in line ]) # extract all the params

        n    = xy.size/2                              # get the number of (x,y) coordines on this line
        x,y  = xy[:n],xy[n:]                          # extract the x and y coordinates from xy
        samples.append(Sample(f,params))                 # create the sample and add to the array
        if pbar: progress_bar()

    samples = trim_samples(np.array(samples),nsamp)
    
    return samples
