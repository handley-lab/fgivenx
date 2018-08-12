---
title: 'fgivenx: a python package for functional posterior plotting'
tags:
  - Python
  - Statistics
  - Bayesian inference
  - Astronomy
authors:
  - name: Will Handley
    orcid: 0000-0002-5866-0445
    affiliation: "1, 2, 3"
affiliations:
 - name: Astrophysics Group, Cavendish Laboratory, J.J.Thomson Avenue, Cambridge, CB3 0HE, UK
   index: 1
 - name: Kavli Institute for Cosmology, Madingley Road, Cambridge, CB3 0HA, UK
   index : 2
 - name: Gonville & Caius College, Trinity Street, Cambridge, CB2 1TA, UK
   index: 3


date: 18 July 2018
bibliography: paper.bib
---

# Summary

Scientists are often concerned with numerical values of parameters in
scientific models. Our knowledge of such things can be quantified and presented
using probability distributions as demonstrated in Figure 1.

![The age and size of the universe, as measured using Planck 2018 data.
(non-Astro)Physicists may have noted that 14 Gigaparsecs is roughly 46 billion
light years. The fact that the observable universe is roughly three times
larger in light years in comparison with its age is explained by the expansion
of space over cosmic history.](planck.png) 

Plots such as Figure 1 can be created using two-dimensional kernel density
estimation using packages such as
[scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
[@scipy], [getdist](http://getdist.readthedocs.io/en/latest/intro.html)
[@getdist], [corner](https://corner.readthedocs.io/en/latest/) [@corner] and
[pygtc](https://pygtc.readthedocs.io/en/latest/) [@pygtc], where the samples
provided as inputs to such programs are typically created by a
Markov-Chain-Monte-Carlo analysis. For further information on MCMC and Bayesian
analysis in general, "Information Theory, Inference and Learning Algorithms" is
highly recommended [@mackay], which is available for free as an online
[pdf](http://www.inference.org.uk/itprnn/book.html)

As well as quantifying the uncertainty of real-valued parameters, scientists
may also be interested in producing a probability distribution for a function
``f(x)``. Take as a universally-relatable case the equation of a straight line
``y = m*x + c``. If one has a probability distribution for the gradient ``m``
and intercept ``c``, then the ability to predict ``y`` knowing ``x``
using the straight line relationship would also be uncertain. The appropriate
plot to make is that of ``P(y|x)`` in the ``(x,y)`` plane, as demonstrated in
Figure 2's bottom right panel.

![An example of plots produced by fgivenx..
Top-left: underlying parameter covariances between ``m`` and ``c`` for
realizations from the prior (blue) and from the posterior (red). 
Top-right realisations function ``y=m*x+c``. 
Bottom-left: The conditional Kullback-Leibler divergence. 
Bottom-right: The probability of measuring y for a given x, essential a contour
version of the panel directly above.
](figure.png) 

``fgivenx`` is a python package for producing plots as in Figure 2, including
the conditional Kullback-Leibler divergence [@Kullback]. This ``y=m*x+c``
example is a little over-simplified, but the code has been used in the latest
Planck papers [@inflation2015][@legacy2018][@inflation2018] to quantify our
knowledge of the primordial power spectrum of curvature perturbations,
in examining the dark energy equation of state [@Hee2015] [@Hee2016] for
measuring errors in parameter estimation [@Higson2017], for providing
diagnostic tests for nested sampling [@Higson2018] and for Bayesian compressive
sensing [@Higson2018b].

``fgivenx`` is a python package for functional posterior plotting, currently
used in astronomy, but will be of use to scientists performing any Bayesian
analysis which has predictive posteriors that are functions. The source code
for ``fgivenx`` is available on
[GitHub](https://github.com/williamjameshandley/fgivenx) and has been archived to
Zenodo with the linked DOI: [@zenodo] 

# Acknowledgements

Contributions and bug-testing were provided by Ed Higson and Sonke Hee.

# References
