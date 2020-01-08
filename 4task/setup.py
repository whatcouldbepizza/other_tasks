from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extension = Extension(
    name="calculations_cython",
    sources=["calculations_cython.pyx"],
    include_dirs=[numpy.get_include()]
)

setup(name="calculations_cython", ext_modules=cythonize([extension]))
