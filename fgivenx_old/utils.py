""" Utilities for computing the probability mass function. """
import scipy.stats
import scipy.interpolate
import numpy

class PMF(object):
    """ Compute the probability mass function.

        The set of samples defines a probability density P(t),
        which is computed using a kernel density estimator.

        From P(t) we define:

                  /
        M(p) =    | P(t) dt
                  / 
              P(t) < p

        This is the cumulative distribution function expressed as a
        function of the probability

        We aim to compute M(t), which indicates the amount of
        probability contained outside the iso-probability contour
        passing through t.


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

        Parameters
        ----------
        samples: List[float]
            Array of samples from a probability density P(t).

        Attributes
        ----------
        lower: float
            lower bound of function f(x|theta) at this x
        upper: float
            upper bound of function f(x|theta) at this x
    """

    def __init__(self, samples):

        # Compute the kernel density estimator from the samples
        kernel = scipy.stats.gaussian_kde(samples)

        # Generate enough samples to get good 2 sigma contours
        n = 10000
        [ts] = kernel.resample(n)

        # Sort the samples in t, and find their probabilities
        ts.sort()
        ps = kernel(ts)

        # Compute the cumulative distribution function M(t) by
        # sorting the ps, and finding the position in that sort
        # We then store this as a log
        logms = numpy.log(scipy.stats.rankdata(ps) / float(n))

        # create an interpolating function of log(M(t))
        self._logpmf = scipy.interpolate.interp1d(ts, logms, bounds_error=False, fill_value=-numpy.inf)
        self.lower = min(ts)
        self.upper = max(ts)

    def __call__(self, t):
        return numpy.exp(self._logpmf(t))
