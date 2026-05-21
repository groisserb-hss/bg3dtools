"""
Rendering utilities for mesh and point cloud visualization.

This package provides wrappers for multiple visualization backends.
Colors and colormaps are exported at the top level. For visualization
functions, import directly from submodules:

    from bg3dtools.render.trimesh import scatt, trisurfsm
    from bg3dtools.render.o3d import scatt, trisurfsm

Submodules
----------
trimesh
    Trimesh-based visualization (scatt, trisurfsm, draw_geometries).
o3d
    Open3D-based visualization (scatt, trisurfsm, draw_line).
matplot
    Matplotlib-based visualization.
colors
    Color utilities and colormaps.
"""

import logging
import multiprocessing as _mp

from .colors import default_colors, get_heatmap_color, xyz_to_rgb

_log = logging.getLogger(__name__)


def run_isolated(fn, *args, timeout=120, **kwargs):
    """Run *fn* in a spawned subprocess to isolate GPU / OpenGL state.

    On macOS, Open3D's Visualizer accumulates Cocoa window-server and
    OpenGL resources across repeated create/destroy cycles.  After enough
    iterations the process deadlocks in C-level code (low CPU, immune to
    Ctrl-C).  Running each render call in its own ``spawn``-ed process
    guarantees a fresh GPU context every time; all handles are released
    when the child exits.

    Parameters
    ----------
    fn : callable
        Must be importable (module-level function) so the ``spawn`` start
        method can pickle it.
    *args, **kwargs
        Forwarded to *fn*.  Must be picklable.
    timeout : float
        Seconds to wait before killing the subprocess.

    Returns
    -------
    bool
        True if the subprocess finished with exit code 0.
    """
    ctx = _mp.get_context('spawn')
    p = ctx.Process(target=fn, args=args, kwargs=kwargs)
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        _log.warning('%s timed out after %ds, killed', fn.__name__, timeout)
        return False
    if p.exitcode != 0:
        _log.warning('%s failed (exit code %d)', fn.__name__, p.exitcode)
        return False
    return True


__all__ = [
    "default_colors",
    "get_heatmap_color",
    "xyz_to_rgb",
    "run_isolated",
]

try:
    from .o3d import render_mesh_to_image, overhead_camera, anterior_camera
    __all__.extend([
        "render_mesh_to_image",
        "overhead_camera",
        "anterior_camera",
    ])
except ImportError:
    render_mesh_to_image = None
    overhead_camera = None
    anterior_camera = None
