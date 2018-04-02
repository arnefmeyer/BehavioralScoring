# -*- coding: utf-8 -*-
"""
Created on Thu May 01 19:44:20 2014

@author: arne
"""

from distutils.core import setup
from os.path import split, join, realpath, exists
from os import chdir

import subprocess
import sys


# Install to local python directory in build inplace?
option = 'install'
if len(sys.argv) > 1:
    option = sys.argv[1]
print "Building option:", option

# Everything is relative to the path of this script
this_dir = split(realpath(__file__))[0]
sub_dirs = []

# Make everything Windows compatible
is_posix = True
if 'win' in sys.platform:
    is_posix = False

# Required to build the package on our HPC cluster
chdir(this_dir)

# Compile Cython code in subpackages
for subdir in sub_dirs:
    dd = join(this_dir, 'lindenlab', subdir)
    setup_file = join(dd, 'setup.py')

    if exists(setup_file):
        print "Trying to build:", setup_file
        if option.lower() == 'build_ext':
            # Just put cython extensions into package subdirectories
            subprocess.call(["/bin/bash", "-i", "-c",
                             "python %s build_ext --inplace" % setup_file])
        else:
            # Install the whole package somewhere
            execfile(setup_file)

if option == 'install':
    setup(
        name='BehavioralScoring',
        packages=['BehavioralScoring'],
        version='0.1',
        description='Behavioral scoring',
        author='Arne F. Meyer',
        author_email='arne.f.meyer@gmail.com',
        install_requires=['numpy', 'scipy', 'scikits-learn', 'opencv',
                          'click', 'tqdm', 'PyQt4', 'matplotlib']
    )
