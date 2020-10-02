from setuptools import setup
from setuptools import find_packages

setup(name='pygcn_kemeny',
      version='0.1',
      description='Graph Convolutional Networks with Kemeny in PyTorch',
      author='Pieke Geraedts',
      author_email='piekegeraedts@hotmail.com',
      url='',
      download_url='https://github.com/PiekeGeraedts/Thesis',
      license='MIT',
      install_requires=['numpy',
                        'torch',
                        'scipy',
                        ],
      package_data={'pygcn_kemeny': ['README.md']},
      packages=find_packages())
