from numpy.random import rand,choice
from numpy import array
from sample import LinearSample
import csv

def read_and_trim(filename,nsamp):

    samples = []

    with open('data.dat', 'r') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            w  = float(row.pop(0))
            xy = array([ float(c) for c in row ])
            n  = xy.size/2
            x  = xy[:n]
            y  = xy[n:]
            if(rand()<w): samples.append(LinearSample(x,y))

    samples = array(samples)
    return choice(samples,nsamp,replace=False)

