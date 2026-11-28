"""IWaVE: estimate flow velocity from waves and patterns in images."""

__version__ = "0.4.0"

import os

# Set this before importing iwave; it is read only once.
# It limits simultaneous fitting processes, not the CPU threads used inside libraries.
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

