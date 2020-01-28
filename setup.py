from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

ext_modules = [Extension('cyTV4D.tv4d_utils', ['cyTV4D/tv4d_utils.pyx'], extra_compile_args=['-fopenmp'], 
	extra_link_args=['-fopenmp', '-lgomp', '-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/9/'])]

os.environ['CC'] = 'gcc-9'

setup(
	name='cyTV4D',
	version='0.0.1',
	author="SE Zeltmann",
	author_email="steven.zeltmann@lbl.gov",
	packages=['cyTV4D'],
    ext_modules=cythonize(ext_modules)
)
