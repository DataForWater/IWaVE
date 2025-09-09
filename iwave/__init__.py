"""IWaVE: Image-based Wave Velocimetry Estimation"""

__version__ = "0.3.1"

import os

CONCURRENCY = os.environ.get("IWAVE_NUM_THREADS", None)
if CONCURRENCY is not None:
    CONCURRENCY = int(CONCURRENCY)

from . import const
from . import dispersion
from . import io
from . import sample_data
from . import spectral
from . import window
from .data_models import LazySpectrumArray, LazyWindowArray
from . import optimise
from .iwave import Iwave

