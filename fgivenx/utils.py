import scipy.stats
import scipy.interpolate
import numpy

from itertools import tee, izip
from scipy.optimize import brentq as root_finder

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def find_all_roots(xs,ys,y,function):

    def f(x):
        return function(x)-y

    roots = []
    for (x0, x1), (y0, y1) in zip(pairwise(xs), pairwise(ys)):
        if y0==y:
            roots.append((x0,y1 > y0))
        elif y1==y:
            roots.append((x1,y1 > y0))
        elif (y0 < y and y1 > y) or (y0 > y and y1 < y):
            x = root_finder(f,x0,x1)
            roots.append((x,y1 > y0))
            
    return roots
            

class PMF(object):

    def __init__(self,samples):
        """ Computes the 'probability mass function' from a set of samples.

            Inputs
            ------
              samples : an array of samples from a probability density P(t)
            Output:
              interpolation function of M(t)


            Explanation
            -----------
            The set of samples defines a probability density P(t), which is
            computed using a kernel density estimator.

            From P(t) we define:
     
                   /
            M(p) = |          P(t) dt
                   / P(t) < p
     
            This is the cumulative distribution function expressed as a
            function of the probability
     
            We aim to compute M(t), which indicates the amount of
            probability contained outside the iso-probability contour passing
            through t
     
     
             ^ P(t)                   ...
             |                     | .   .
             |                     |       .
            p|- - - - - - - - - - .+- - - - . - - - - - - - - - - - 
             |                   .#|        #.
             |                  .##|        ##.
             |                  .##|        ##.
             |                 .###|        ###.       M(p) is the shaded area
             |                 .###|        ###.
             |                 .###|        ###.
             |                .####|        ####.
             |                .####|        ####.
             |              ..#####|        #####..
             |          ....#######|        #######....
             |         .###########|        ###########.
             +---------------------+-------------------------------> t 
                                  t

             ^ M(p)                        ^ M(t)                   
             |                             |                        
            1|                +++         1|         + 
             |               +             |        + +
             |       ++++++++              |       +   +            
             |     ++                      |     ++     ++          
             |   ++                        |   ++         ++        
             |+++                          |+++             +++     
             +---------------------> p     +---------------------> t
            0                   1                                   
        """
        # get a sense of the center and scale of the PDF
        mu = numpy.mean(samples)
        sigma = numpy.std(samples)

        # Compute the kernel density estimator from the samples
        kernel = scipy.stats.gaussian_kde(samples)
        #ts = sorted(kernel.resample(size=10000)[0])
        ts = sorted(samples)
        ps = kernel(ts)
        n = float(len(ts))

        # Sort the grid by probability
        sort_by_p = sorted([(p,t) for p, t in zip(ps,ts) ])

        # Compute the cumulative distribution function
        cdf = [(t,(i+1)/n) for i,(_,t) in enumerate(sort_by_p)]
        cdf = sorted(cdf)

        # define the function
        ts = [t for t,_ in cdf]
        logms = [numpy.log(m) for _,m in cdf]
        self.pmf = scipy.interpolate.interp1d(ts, logms, bounds_error=False, fill_value=-numpy.inf)
        self.lower = min(ts)
        self.upper = max(ts)

        #import matplotlib.pyplot as plt
        #fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
        #ax1.plot(ts,ps)
        #ax2.plot(ts,numpy.exp(logms))
        #plt.show()

    def __call__(self,t):
        return numpy.exp(self.pmf(t))



