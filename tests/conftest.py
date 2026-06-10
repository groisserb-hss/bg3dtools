"""Pytest configuration.

The open3d import below is load-bearing: on macOS, importing torch before
open3d segfaults the process (OpenMP runtime clash, "OMP: Error #179"),
while the reverse order is fine. Importing open3d here — before any test
module pulls in torch — keeps the suite runnable with a bare `pytest tests/`.
open3d is an optional dependency, so its absence is tolerated.
"""

try:
    import open3d  # noqa: F401
except ImportError:
    pass
