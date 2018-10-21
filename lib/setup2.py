from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import platform

ext_modules=[
    Extension("gmr_cython",
              ["gmr_cython.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args = ['-fopenmp']
              ) 
]

setup( 
  name = "gmr_cython",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)
