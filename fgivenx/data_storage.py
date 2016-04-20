""" Objects for holding posteriors and samples.
"""
import numpy.random

class Sample(object):
    """ Posterior sample.

        Parameters
        ----------
        params: List[float]
            The sampled parameters.
        paramnames: List[str]
            The names of each parameter, in the same order as params.
        logL: float
            The log-Likelihood of the sample.
        w: float
            The posterior weight of the sample. Need not be normalised.


        Attributes
        ----------
        params: List[float]
            The sampled parameters.
        logL: float
            The log-Likelihood of the sample.
        w: float
            The posterior weight of the sample. Need not be normalised.
    """

    def __init__(self, params, paramnames, logL, w=1):
        self.params = params
        self._params_from_name = {p:t for p, t in zip(paramnames, params)}
        self._params_from_name['logL'] = logL
        self.logL = logL
        self.w = w

    def __getitem__(self, paramname):
        """ Return item value from paramname

            Parameters
            ----------
            paramname: str
                The name of the desired parameter

            Returns
            -------
            float
                Value of the parameter with the given paramname.
        """
        return self._params_from_name[paramname]

    def __str__(self):
        return "w=%s, logL=%s, params=%s" % (self.w, self.logL, self.params)

    def __repr__(self):
        return self.__str__()


class Posterior(object):
    """ A set of posterior samples """

    def __init__(self, chains_file, paramnames_file):
        """ Initialise from a chains_file and a paramnames file """

        # load the paramnames
        self.paramnames = []
        for line in open(paramnames_file, 'r'):
            line = line.split()
            paramname = line[0]
            self.paramnames.append(paramname)

        self.samples = []
        for line in open(chains_file, 'r'):
            line = line.split()
            w, logL, params = float(line[0]), float(line[1]), [float(t) for t in line[2:]]
            sample = Sample(params, self.paramnames, logL, w)
            self.samples.append(sample)

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def trim_samples(self, nsamp=None):
        """ Trim samples """

        maxw = max([s.w for s in self])

        trimmed_samples = []
        for s in self:
            if numpy.random.rand() < s.w/maxw:
                s.w = 1.0
                trimmed_samples.append(s)

        if nsamp is not None:
            if nsamp < len(trimmed_samples):
                trimmed_samples = list(numpy.random.choice(trimmed_samples, nsamp))

        self.samples = trimmed_samples
        return self


class FunctionSample(Sample):
    """ Functional posterior sample

        This is derived from the Sample class.
        Please look there for initialisers and attributes.
    """

    def set_function(self, f, chosen_parameters):
        """ Set the function to be evaluated, using the chosen parameters

            Parameters
            ----------
            f: function
                Parameters
                ----------
                x: float
                    independent variable.
                theta: List[float]
                    parameters that the function depends on.

                Returns
                -------
                float
                    The value of f(x|theta).

            chosen_parameters: List[str]
                The names of the parameters from sample that the function uses
        """

        params = [self[p] for p in chosen_parameters]
        self._function = lambda x, p=params: f(x, p)

    def __call__(self, x):
        """ Value of f(x|theta), with theta equal to the sample parameters. """
        return self._function(x)


class FunctionalPosterior(Posterior):
    """ A posterior containing functions """

    def set_function(self, function, chosen_parameters):
        """ Load the function into each of the posteriors """
        for sample in self:
            sample.__class__ = FunctionSample
            sample.set_function(function, chosen_parameters)

        return self

    def __call__(self, x):
        """ Return a set of samples of the functions value at x.

            Parameters
            ----------
            x: float
                value of x to evaluate functions at.

            Returns
            -------
            List[float]
                The value of f at specified x for every sample.
        """
        return [s(x) for s in self]
