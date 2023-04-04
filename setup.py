#!/usr/bin/env python3

try:
    from setuptools import setup, Command
except ImportError:
    from distutils.core import setup, Command


def readme():
    with open('README.rst') as f:
        return f.read()


def get_version(short=False):
    with open('README.rst') as f:
        for line in f:
            if ':Version:' in line:
                ver = line.split(':')[2].strip()
                if short:
                    subver = ver.split('.')
                    return '%s.%s' % tuple(subver[:2])
                else:
                    return ver


setup(name='fgivenx',
      version=get_version(),
      description='fgivenx: Functional Posterior Plotter',
      long_description=readme(),
      author='Will Handley',
      author_email='wh260@cam.ac.uk',
      url='https://github.com/fgivenx/fgivenx',
      packages=['fgivenx', 'fgivenx.test'],
      install_requires=['matplotlib', 'numpy', 'scipy'],
      setup_requires=['pytest-runner'],
      extras_require={
          'docs': ['sphinx', 'sphinx_rtd_theme', 'numpydoc'],
          'parallel': ['joblib'],
          'progress_bar': ['tqdm'],
          'getdist_chains': ['getdist']
          },
      tests_require=['pytest', 'pytest-mpl'],
      include_package_data=True,
      license='MIT',
      classifiers=[
                   'Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'Natural Language :: English',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Topic :: Scientific/Engineering :: Visualization',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: Scientific/Engineering :: Mathematics',
      ],
      )
