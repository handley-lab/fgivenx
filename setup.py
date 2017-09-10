#!/usr/bin/env python3

from distutils.core import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='fgivenx',
      version='1.0.13',
      author='Will Handley',
      author_email='wh260@cam.ac.uk',
      url='https://github.com/williamjameshandley/fgivenx',
      packages=['fgivenx'],
      install_requires=['numpy','matplotlib','scipy','joblib','mpi4py','tqdm'],
      license='MIT',
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
      long_description=readme(),
      )
