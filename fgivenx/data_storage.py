class Sample(object):
    """ class to store posterior sample """
    def __init__(self, params, logL, w=1):
        """ Initialise with params, loglikelihood and optional weight 
        
            params: dictionary of { paramname:value }
            logL: float
            w: float
        
        """
        self.params = params
        self.logL = logL
        self.w = w

class FunctionSample(object):
    """ class to store functional posterior sample """
    def __init__(self, f, logL, w=1):
        """ Initialise with params, loglikelihood and optional weight 
        
            f: function
            w: float
        
        """
        self.f = f
        self.logL = logL
        self.w = w

    def __call__(self,x):
        return self.f(x)

def load_posterior(chains_file, paramnames_file):
        
    # load the paramnames
    paramnames = []
    for line in open(paramnames_file,'r'):
        line = line.split()
        paramname = str(line.pop(0))
        paramnames.append(paramname)

    samples = []
    for line in open(chains_file,'r'):
        line   = line.split()               
        w, logL, params = float(line[0]), float(line[1]), [float(t) for t in line[2:]]
        params = { p:t for p, t in zip(paramnames, params) } # extract all the params
        samples.append(Sample(params,logL,w))

    return samples, paramnames

def create_functional_posterior(posterior, function, chosen_parameters):
    functional_posterior = []

    for sample in posterior:
        params = [sample.params[p] for p in chosen_parameters]
        f = lambda x,p=params: function(x, p)
        logL = sample.logL
        w = sample.w
        function_sample = FunctionSample(f,logL,w)

        functional_posterior.append(function_sample)

    return functional_posterior


