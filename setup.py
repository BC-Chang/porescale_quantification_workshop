import numpy as np

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("src.subset_analysis", sources=["./src/*.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'])
]
setup(
    name="subset_analysis",
    ext_modules=cythonize(extensions),
)