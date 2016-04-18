import numpy.random

class Sample(object):
    """ class to store posterior sample """
    def __init__(self, params, paramnames, logL, w=1):
        """ Initialise with params, loglikelihood and optional weight 
        
            params: dictionary of { paramname:value }
            logL: float
            w: float
        
        """
        self.params = params
        self.params_from_name = { p:t for p, t in zip(paramnames, params) }
        self.params_from_name['logL'] = logL
        self.logL = logL
        self.w = w

    def __getitem__(self,paramname):
        """ Return item value from paramname """
        return self.params_from_name[paramname]
    def __str__(self):
        """ String """
        return "w=%s, logL=%s, params=%s" % (self.w, self.logL, self.params)
    def __repr__(self):
        """ Representation """
        return self.__str__()

class FunctionSample(Sample):
    """ class to store functional posterior sample """

    def set_function(self,function,chosen_parameters):
        """ Set the function to be evaluated, using the chosen parameters"""

        params = [self[p] for p in chosen_parameters]
        self.f = lambda x, p=params: function(x, p)

    def __call__(self,x):
        """ return parameter with name """
        return self.f(x)


class Posterior(object):
    """ A set of posterior samples """

    def __init__(self,chains_file,paramnames_file):
        """ Initialise from a chains_file and a paramnames file """

        # load the paramnames
        self.paramnames = []
        for line in open(paramnames_file,'r'):
            line = line.split()
            paramname = line[0]
            self.paramnames.append(paramname)

        self.samples = []
        for line in open(chains_file,'r'):
            line   = line.split()               
            w, logL, params = float(line[0]), float(line[1]), [float(t) for t in line[2:]]
            sample = Sample(params,self.paramnames,logL,w)
            self.samples.append(sample)

    def __iter__(self):
        """ Iterate through the samples """
        return iter(self.samples)

    def __len__(self):
        """ Number of samples """
        return len(self.samples)

    def trim_samples(self,nsamp=None):
        """ Trim samples """

        n = len(self)
        maxw = max([s.w for s in self])

        trimmed_samples = []
        for s in self:
            if numpy.random.rand() < s.w/maxw:
                s.w = 1.0
                trimmed_samples.append(s)


        if nsamp is not None:
            if nsamp < len(trimmed_samples):
                trimmed_samples = list(numpy.random.choice(trimmed_samples,nsamp))

        self.samples = trimmed_samples
        print "Samples trimmed from " , n, " to ", len(self)


class FunctionalPosterior(Posterior):
    """ A posterior containing functions """

    def set_function(self,function,chosen_parameters):
        """ Load the function into each of the posteriors """
        for sample in self:
            sample.__class__ = FunctionSample
            sample.set_function(function,chosen_parameters)

        return self
