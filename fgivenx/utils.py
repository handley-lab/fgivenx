import scipy.stats
import scipy.interpolate
import numpy

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
            The set of samples defines a probability density P(t), 
            which is computed using a kernel density estimator.

            From P(t) we define:
     
                   /
            M(p) = |          P(t) dt
                   / P(t) < p
     
            This is the cumulative distribution function expressed as a
            function of the probability
     
            We aim to compute M(t), which indicates the amount of
            probability contained outside the iso-probability contour
            passing through t
     
     
             ^ P(t)                   ...
             |                     | .   .
             |                     |       .
            p|- - - - - - - - - - .+- - - - . - - - - - - - - - - - 
             |                   .#|        #.
             |                  .##|        ##.
             |                  .##|        ##.
             |                 .###|        ###.     M(p) 
             |                 .###|        ###.     is the 
             |                 .###|        ###.     shaded area
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

        # Compute the kernel density estimator from the samples
        kernel = scipy.stats.gaussian_kde(samples)
        ts = kernel.resample(10000)[0]
        ps = kernel(ts)

        n = float(len(ts))

        # Sort the grid by probability
        sort_by_p = [(p,t) for p, t in zip(ps,ts)]
        sort_by_p.sort()

        # Compute the cumulative distribution function
        cdf = [(t,(i+1)/n) for i,(_,t) in enumerate(sort_by_p)]
        cdf.sort()

        # define the function
        self.pmf = scipy.interpolate.interp1d(
                [t for t,_ in cdf],
                [numpy.log(m) for _,m in cdf],
                bounds_error=False, 
                fill_value=-numpy.inf
                )
        self.lower = min(ts)
        self.upper = max(ts)

    def __call__(self,t):
        return numpy.exp(self.pmf(t))



