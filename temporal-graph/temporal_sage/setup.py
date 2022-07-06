from setuptools import setup, Extension
from torch.utils import cpp_extension

import os.path as osp

# NEBULA_INCLUDE=osp.join(osp.dirname(osp.abspath(__file__)), "include")
# NEBULA_LIBRARY=osp.join(osp.dirname(osp.abspath(__file__)), "lib")

NEBULA_ROOT=osp.expanduser("~/.local/nebula")
NEBULA_INCLUDE=osp.join(NEBULA_ROOT, "include")
NEBULA_LIBRARY=osp.join(NEBULA_ROOT, "lib")

setup(
    name="query_graph",
    ext_modules=[
        cpp_extension.CppExtension(
            name="query_graph",
            sources=["query_graph.cpp"],
            include_dirs=[NEBULA_INCLUDE],
            library_dirs=[NEBULA_LIBRARY],
            libraries=["nebula_graph_client"],
        ),
    ],
    cmdclass={
        "build_ext": cpp_extension.BuildExtension,
    }
)
