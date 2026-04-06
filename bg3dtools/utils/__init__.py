"""
General utilities for bg3dtools.

Provides timing, scheduling, array manipulation, filesystem operations,
statistical analysis, and display utilities.
"""

# Display utilities (must be first for conditional imports)
from .display import matplotlib_is_headless

# Scheduling and timing
from .scheduling import (
    AverageMeter,
    Timer,
    ConvergenceScheduler,
    wait_for_completion,
    HiddenPrints,
)

# NumPy helpers
from .np_helpers import (
    row_normalize,
    pad_to_size,
    truncated_normal,
    is_valid_array,
)

# Algorithms
from .algorithms import (
    argsort,
    farthest_point_sampling,
    save_pca,
    load_pca,
)

# Filesystem utilities
from .fs import zip_file
from .fs_mutex import acquire_mutex, release_mutex

# Image utilities
from .image import auto_canny

# Statistical utilities
from .stats import (
    icc_value,
    perm_test_icc_diff_arrays,
)

# Output suppression
from .suppress import SuppressCppStderr, suppress_cpp_stderr

__all__ = [
    # display
    "matplotlib_is_headless",
    # scheduling
    "AverageMeter",
    "Timer",
    "ConvergenceScheduler",
    "wait_for_completion",
    "HiddenPrints",
    # np_helpers
    "row_normalize",
    "pad_to_size",
    "truncated_normal",
    "is_valid_array",
    # algorithms
    "argsort",
    "farthest_point_sampling",
    "save_pca",
    "load_pca",
    # fs
    "zip_file",
    "acquire_mutex",
    "release_mutex",
    # image
    "auto_canny",
    # stats
    "icc_value",
    "perm_test_icc_diff_arrays",
    # suppress
    "SuppressCppStderr",
    "suppress_cpp_stderr",
]

# Optional: plotting (requires GUI)
if not matplotlib_is_headless():
    from .plotting import bland_altman
    __all__.append("bland_altman")
