from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("src.find_interval", ["./src/*.pyx"],
              include_dirs=[numpy.get_include()]),
]
setup(
    name="find_interval",
    ext_modules=cythonize(extensions),

)


