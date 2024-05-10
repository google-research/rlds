# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
"""Install script for setuptools."""

import datetime
from importlib import util as import_util
import sys

from setuptools import find_packages
from setuptools import setup

spec = import_util.spec_from_file_location('rlds_version', 'rlds_version.py')
rlds_version = import_util.module_from_spec(spec)
spec.loader.exec_module(rlds_version)

requirements = [
    'absl-py',
    'numpy',
]

# TF is a requirement, but some of the libraries that depend on us require
# a specific version of TF, so we let them install it.
optional_requirements = [
    'tensorflow',
    'tensorflow-datasets',
    'dm-reverb'
]

long_description = """RLDS is a library to manipulate datasets with episodic
structure. When data is loaded as a dataset of episodes containing nested
datasets of steps, RLDS provides utils to manipulate them and makes it easier
to perform transformations that are common in RL, Apprenticeship Learning or
other algorithms that learn from experience

For more information see [github repository](http://github.com/google-research/rlds)."""

# Get the version from metadata.
version = rlds_version.__version__

# If we're releasing a nightly/dev version append to the version string.
if '--nightly' in sys.argv:
  sys.argv.remove('--nightly')
  version += '.dev' + datetime.datetime.now().strftime('%Y%m%d')

setup(
    name='rlds',
    version=version,
    description='A Python library for Reinforcement Learning Datasets.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Google Research',
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning python machine learning datasets',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'tensorflow': optional_requirements,
        },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
