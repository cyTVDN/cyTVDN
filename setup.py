from setuptools import setup, Extension

# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Build import cythonize
import platform
import os
from glob import glob

if platform.system() == "Windows":
    extra_compile_args = ["/openmp"]
    extra_link_args = ["/openmp"]
elif platform.system() == "Darwin":
    extra_link_args = ["-fopenmp=libomp"]
    extra_compile_args = ["-fopenmp=libomp"]

    # we are on a Mac, link to the Homebrew installation of llvm
    extra_link_args.append(
        "-L" + "/usr/local/opt/llvm/lib"
    )

    # # Previously, I used Homebrew-installed gcc...
    # # However, it has been giving me unexpected behavior in parallel code
    # os.environ["CC"] = "gcc-11"

    # # LLVM Clang seems to work correctly!
    # # If CC and LDSHARED are not set in the envoronment, try to find Homebrew LLVM clang...
    if "CC" not in os.environ:
        os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
    # if "LDSHARED" not in os.environ:
    #     os.environ["LDSHARED"] = (
    #         "/usr/local/opt/llvm/bin/clang -bundle"
    #     )

    # Problems with Python.h not finding <stdio.h>?
    # see: https://stackoverflow.com/questions/58278260/cant-compile-a-c-program-on-a-mac-after-upgrading-to-catalina-10-15/58323411#58323411
else:
    # if we are on linux, check if we are using intel or Cray/gcc
    if "CC" not in os.environ:
        extra_compile_args = ["-fopenmp"]
        extra_link_args = ["-fopenmp"]
    elif "icc" in os.environ["CC"]:
        extra_compile_args = ["-qopenmp"]
        extra_link_args = ["-qopenmp"]
    else:
        extra_compile_args = ["-fopenmp"]
        extra_link_args = ["-fopenmp"]


ext_modules = [
    Extension(
        "cyTVDN.utils",
        ["cyTVDN/utils.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "cyTVDN.anisotropic",
        ["cyTVDN/anisotropic.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "cyTVDN.halfisotropic",
        ["cyTVDN/halfisotropic.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="cyTVDN",
    version="0.1",
    author="SE Zeltmann",
    author_email="steven.zeltmann@lbl.gov",
    packages=["cyTVDN"],
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
    install_requires=[
        "Cython", 
        "hurry.filesize", 
        "psutil", 
        "tabulate", 
        "loguru", 
        "numpy",
        "ipython",
        "tqdm"],
    extras_require={"MPI": ["mpi4py", "h5py"], "IO":["py4DSTEM", "ncempy"]},
    setup_requires=["Cython"],
    entry_points={"console_scripts": ["cyTVMPI=cyTVDN.mpi:run_MPI"]},
)
