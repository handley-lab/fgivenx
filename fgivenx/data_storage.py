""" Objects for holding posteriors and samples.
"""
import numpy.random
import pickle

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


    def __init__(self, line, paramnames):
        """ Initialise a posterior sample from a line from a getdist file.

            Lines in a getdist file are typically:
                weight -2*logL param1 param2 ... paramN

            Parameters
            ----------
            line: str
                Pure getdist line.
            paramnames: List[str]
                List of parameter names.
        """
        line = line.split()
        self.w = float(line[0])
        self.logL = float(line[1])/(-2.0)
        self.params = [float(t) for t in line[2:]]
        self._params_from_name = {p:t for p, t in zip(paramnames, self.params)}
        self._params_from_name['logL'] = self.logL


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


class FunctionSample(Sample):
    """ Functional posterior sample

        This is derived from the Sample class, but adds a function
        f(x|theta) where theta is some subset of the posterior sample
        parameters.  Please look there for initialisers and
        attributes.
    """
    def __init__(self, line, paramnames):
        super(FunctionSample, self).__init__(line, paramnames)
        self._f = None

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

        self.params = [self[p] for p in chosen_parameters]
        self._f = f

        return self

    def __call__(self, x):
        """ Value of f(x|theta), with theta equal to the sample parameters. """
        return self._f(x,self.params)



class Posterior(list):
    """ A set of posterior samples.

        Container-style iterable object derived from list.

        Parameters
        ----------
        chains_file: str
            Name of the file where the posterior samples are stored.
            These should be a text data file with columns:
                weight  log-likelihood  <parameters>
            Typically this file is produced by getdist.
        paramnames_file: str
            Where the names of the the parameters are stored. This
            should be a text file with one parameter name per line
            (no spaces), in the order they appear in chains_file.
            Typically this file is produced by getdist.
    """

    def __init__(self, chains_file=None, paramnames_file=None, SampleType=Sample):
        # Call the list initialiser
        super(Posterior, self).__init__()

        # load the paramnames
        if paramnames_file is not None:
            paramnames = [line.split()[0] for line in open(paramnames_file, 'r') ]

        # Load the list
        if chains_file is not None:
            for line in open(chains_file, 'r'):
                self.append(SampleType(line, paramnames))

    def trim_samples(self, nsamp=None):
        """ Trim samples.

            Thins weighted samples to an equally weighted set, and
            then further thins to nsamp if specified.

            Parameters
            ----------
            nsamp: int, optional:
                The number of posterior samples to be kept
        """

        # Find the max weight
        maxw = max([s.w for s in self])
        new_samples = type(self)()

        # delete each sample with a probability w/maxw
        for s in self:
            if numpy.random.rand() < s.w/maxw:
                s.w = 1.0
                new_samples.append(s)

        # Remove any more at random we still need to be lower
        if nsamp is not None:
            numpy.random.shuffle(new_samples)
            del new_samples[nsamp:]

        self[:] = new_samples[:]
        return self

    def normalise(self, evidence=1.0):
        """ Normalise so that the samples sum to evidence
        """
        wtot = sum([s.w for s in self])
        for s in self:
            s.w *= evidence/wtot

        return self

    @classmethod
    def load(cls, filename):
        """ Factory function to load posteriors from pickle file.

            Parameters
            ----------
            filename: str
                The name of the file.
        """
        return pickle.load(open(filename, 'rb'))


    def save(self, filename):
        """ Save posteriors to pickle file.

            Parameters
            ----------
            filename: str
                The name of the file.
        """
        pickle.dump(self, open(filename, 'wb'))
        return self

    

class FunctionalPosterior(Posterior):
    """ A posterior containing FunctionSample s """

    def __init__(self, chains_file=None, paramnames_file=None):
        super(FunctionalPosterior, self).__init__(chains_file, paramnames_file, FunctionSample)


    def set_function(self, function, chosen_parameters):
        """ Load the function into each of the posteriors """
        for sample in self:
            sample.set_function(function, chosen_parameters)
        return self

    def __call__(self, x):
        """ Return a set of samples of the functions value at x.

            i.e. returns { f(x|theta) for theta in samples }

            Parameters
            ----------
            x: float
                value of x to evaluate functions at.

            Returns
            -------
            List[float]
                The value of f at specified x for every sample.
        """
        return [f(x) for f in self]
