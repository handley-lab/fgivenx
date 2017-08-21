#!/usr/bin/env python3

from distutils.core import setup

setup(name='fgivenx',
      version='1.0',
      author='Will Handley',
      author_email='wh260@cam.ac.uk',
      url=None,
      packages=['fgivenx'],
      install_requires=['numpy','pickle','matplotlib','scipy','joblib','mpi4py'],
      license='MIT'
      classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Natural Language :: English',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3.6',
      'Topic :: Scientific/Engineering :: Astronomy',
      'Topic :: Scientific/Engineering :: Physics',
      'Topic :: Scientific/Engineering :: Visualization',
      'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      description='Functional Posterior Plotter',
      long_description=
      """ Functional Posterior Plotter.

    These packages allows one to compute a predictive posterior of a function,
    dependent on sampled parameters. We assume one has a Bayesian posterior
    Post(theta|D,M) described by a set of posterior samples {theta_i}~Post. If
    there is a a function parameterised by theta f(x;theta), then this script
    will produce a contour plot of the conditional posterior P(f|x,D,M) in the
    (x,f) plane.
    
    The driving routine is fgivenx.compute_contours.
    """,
      )

