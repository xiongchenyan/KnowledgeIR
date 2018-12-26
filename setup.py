'''
/*==========================================================================
 * Copyright (c) 2018 Carnegie Mellon University.  All Rights Reserved.
 *
 * Use of the Lemur Toolkit for Language Modeling and Information Retrieval
 * is subject to the terms of the software license set forth in the LICENSE
 * file included with this software, and also available at
 * http://www.lemurproject.org/license.html
 *
 *==========================================================================
*/
'''

from setuptools import setup
from setuptools import find_packages


setup(name='knowledge4ir',
      version='0.0',
      description='knowledge base 4 ir',
      author='Chenyan Xiong',
      install_requires=['scikit-learn', 'sklearn', 'numpy', 'scipy', 'traitlets', 'six',
                        'google-api-python-client', 'elasticsearch', 'nltk', 'keras',
                        'gensim', 'rdflib', 'pyahocorasick', 'torch'
                        ],
      packages=find_packages()
      )
