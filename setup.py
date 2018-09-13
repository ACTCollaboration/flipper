from distutils.core import setup, Extension
import os


setup(name='flipper',
      version='0.1',
      description='Map manipulation tools',
      author='Sudeep Das, Tobias Marriage',
      packages=['flipper'],
      install_requires=['healpy>=1.10.3',
                        'numpy>=1.11.3',
                        'astropy>=1.3',
                        'matplotlib>=1.5.3',
                        'scipy>=0.18.1',
                        'astLib>=0.8.0'],
      zip_safe=False)
