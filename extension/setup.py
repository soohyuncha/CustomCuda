from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
        name='linear_custom',
        ext_modules=[
            CUDAExtension('linear_custom', ['linear.cpp', 'linear_kernel.cu'])
        ],
        cmdclass={'build_ext': BuildExtension}
        )
