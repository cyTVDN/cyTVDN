from setuptools import setup, Extension
# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Build import cythonize
import platform
import os

if platform.system() == 'Windows':
    extra_compile_args = ['/openmp']
    extra_link_args = ['/openmp']
elif platform.system() == "Darwin":
    extra_link_args = ['-fopenmp']
    extra_compile_args = ['-fopenmp']

    # we are on a Mac, link to the Homebrew installation of llvm
    extra_link_args.append('-lgomp')
    extra_link_args.append('-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/9/')
    # use the Homebrew gcc
    os.environ['CC'] = 'gcc-9'
else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

# NOTE: On NERSC Cori things are harder:
# What worked for me is to run `module swap PrgEnv-intel PrgEnv-cray`
# to use the Cray compilers and also specify the compiler
# when running setup.py: `CC='cc' LDSHARED='cc -shared' python setup.py build_ext`

ext_modules = [Extension(
    'cyTV4D.utils', ['cyTV4D/utils.pyx'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args),
    Extension(
        'cyTV4D.anisotropic', ['cyTV4D/anisotropic.pyx'],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)]

setup(
    name='cyTV4D',
    version='0.0.1',
    author="SE Zeltmann",
    author_email="steven.zeltmann@lbl.gov",
    packages=['cyTV4D'],
    ext_modules=cythonize(ext_modules),
    install_requires=[
        'Cython',
        'hurry.filesize',
        'psutil',
        'tabulate'
    ],
    setup_requires=['Cython']
)
