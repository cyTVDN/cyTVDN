from setuptools import setup, Extension
# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Build import cythonize
import platform
import os

if platform.system() == 'Windows':
    extra_compile_args = ['/openmp']
    extra_link_args = ['/openmp']
else:
    extra_link_args = ['-fopenmp']
    extra_compile_args = ['-fopenmp']

if platform.system() == 'Darwin':
    # we are on a Mac, link to the Homebrew installation of llvm
    extra_link_args.append('-lgomp')
    extra_link_args.append('-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/9/')
    # use the Homebrew gcc
    os.environ['CC'] = 'gcc-9'

ext_modules = [Extension(
    'cyTV4D.tv4d_utils', ['cyTV4D/tv4d_utils.pyx'],
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
        'psutil'
    ]
)
