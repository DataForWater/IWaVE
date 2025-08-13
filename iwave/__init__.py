"""IWaVE: Image-based Wave Velocimetry Estimation"""

__version__ = "0.2.0"

import os

from . import const
from . import dispersion
from . import io
from . import sample_data
from . import spectral
from . import window
from .data_models import LazySpectrumArray, LazyWindowArray
from . import optimise
from .iwave import Iwave

# Limit thread usage for OpenMP and BLAS backends
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS threads
os.environ["MKL_NUM_THREADS"] = "1"  # MKL threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr threads
