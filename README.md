Functional Posterior Plotter
----------------------------

These packages allows one to compute a predictive posterior of a function,
dependent on sampled parameters. We assume one has a Bayesian posterior
Post(theta|D,M) described by a set of posterior samples {theta_i}~Post. If
there is a a function parameterised by theta f(x;theta), then this script will
produce a contour plot of the conditional posterior P(f|x,D,M) in the (x,f)
plane.

The driving routine is `fgivenx.compute_contours`, and example usage can be
found by running `help(fgivenx)`.

