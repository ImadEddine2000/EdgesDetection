from setuptools import setup, find_packages
setup(name='A Lightweight Neural Network for Fast and Interactive Edge Detection in 3D Point Clouds',
      version='3.0',
      packages=find_packages(),
      install_requires=['pandas', 'matplotlib', 'pydot', 'graphviz' ,'numba==0.57.1', 'numpy==1.24', 'tqdm', 'tensorflow==2.10' ,'wheel'])