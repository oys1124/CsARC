"""Build the optional CSA C++ range coder in place.

Run from the project root:
    python codec/setup_arithmetic_fast.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="csa-arithmetic-fast",
    ext_modules=[
        CppExtension(
            name="codec._arithmetic_fast",
            sources=["codec/csrc/arithmetic_fast.cpp"],
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
