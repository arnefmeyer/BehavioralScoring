#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

from distutils.core import setup

setup(
    name='BehavioralScoring',
    packages=['BehavioralScoring'],
    version='0.1',
    description='Simple behavioral scoring using accelerometer data',
    author='Arne F. Meyer',
    author_email='arne.f.meyer@gmail.com',
    install_requires=['numpy',
                      'scipy',
                      'scikits-learn',
                      'opencv',
                      'click',
                      'tqdm',
                      'PyQt4',
                      'matplotlib']
)
