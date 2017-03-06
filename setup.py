from setuptools import setup
from setuptools import find_packages


setup(name='knowledge4ir',
      version='0.0',
      description='knowledge base 4 ir',
      author='Chenyan Xiong',
      install_requires=['scikit-learn', 'sklearn', 'numpy', 'scipy', 'traitlets', 'six',
                        'google-api-python-client', 'elasticsearch', 'nltk', 'keras',
                        'gensim', 'rdflib'
                        ],
      packages=find_packages()
      )