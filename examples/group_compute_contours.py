""" Compute contours.

    Things to specify:

    xmin: float
        minimum of x range
    xmax: float
        maximum of x range
    chains_file: str
        Name of the file where the posterior samples are stored. These should
        be a text data file with columns:
            weight  log-likelihood  <parameters>
        Typically this file is produced by getdist.
    paramnames_file: str
        Where the names of the the parameters are stored. This should be a text
        file with one parameter name per line (no spaces), in the order they
        appear in chains_file.
        Typically this file is produced by getdist.

    f: function
        The function to be inferred f(x;theta)
        Parameters
        ----------
        x: float
            independent variable
        theta: List[float]
            parameters of function

    chosen_parameters: List[str]
        The parameter names that the function depends on, in the order f as
        determined by f.
"""
from fgivenx.contours import Contours
from fgivenx.data_storage import FunctionalPosterior

# setup
# -----
xmin = -5
xmax = 5
chains_file = 'chains/test.txt'              # posterior files
paramnames_file = 'chains/test.paramnames'   # paramnames  file

def f(x, theta):
    """ Simple y = m x + c """
    m, c = theta

    return m * x + c

choices = [['m_' + str(i), 'c_' + str(i)] for i in range(1, 5)]


# Computing contours
# ------------------
# load the posteriors from file, and trim them
nsamp = 500
posterior = FunctionalPosterior(chains_file, paramnames_file)
trimmed_posterior = posterior.trim_samples(nsamp)

for i, chosen_parameters in enumerate(choices):
    # set the function
    trimmed_posterior.set_function(f, chosen_parameters)

    # Create the contours and save
    contourfile = 'contours/posterior' + str(i) + '.pkl'
    message = "(" + str(i+1) + "/4) Computing contours for f(x|" + ','.join(chosen_parameters) + ")"
    contours = Contours(trimmed_posterior, (xmin, xmax), message=message).save(contourfile)
