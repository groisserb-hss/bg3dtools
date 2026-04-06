"""
Visualization and platform utilities for spectral matching.

Provides side-by-side mesh plotting, vertex coloring helpers, and
platform detection for GPU/CPU selection.
"""

# Derived from pyFM by Robin Magnet (MIT License) — see /THIRD_PARTY_NOTICES.txt
import platform
import sysconfig
import sys
import numpy as np
from spectral_match.tools.mesh_class import Mesh


def double_plot(
    myMesh1: Mesh,
    myMesh2: Mesh,
    cmap1: np.ndarray | None = None,
    cmap2: np.ndarray | None = None,
) -> None:
    from bg3dtools.render.trimesh import trisurfsm, draw_geometries
    v1, f1 = myMesh1.v, myMesh1.f
    v2, f2 = myMesh2.v, myMesh2.f
    o = max(np.max(v1[:, 0]), np.max(v2[:, 0])) - min(np.min(v1[:, 0]), np.min(v2[:, 0]))
    m1 = trisurfsm(v1 + [o, 0, 0], f1, cmap1, render=False)
    m2 = trisurfsm(v2 - [o, 0, 0], f2, cmap2, render=False)
    draw_geometries([m1, m2])


def vcolor(vertices: np.ndarray) -> np.ndarray:
    min_coord = np.min(vertices, axis=0, keepdims=True)
    max_coord = np.max(vertices, axis=0, keepdims=True)
    cmap = (vertices - min_coord) / (max_coord - min_coord)
    return cmap


def get_platform() -> str:
    """Return a string with current platform (system and machine architecture).

    This attempts to improve upon `sysconfig.get_platform` by fixing some
    issues when running a Python interpreter with a different architecture than
    that of the system (e.g. 32bit on 64bit system, or a multiarch build),
    which should return the machine architecture of the currently running
    interpreter rather than that of the system (which didn't seem to work
    properly). The reported machine architectures follow platform-specific
    naming conventions (e.g. "x86_64" on Linux, but "x64" on Windows).

    Example output strings for common platforms:

        darwin_(ppc|ppc64|i368|x86_64|arm64)
        linux_(i686|x86_64|armv7l|aarch64)
        windows_(x86|x64|arm32|arm64)

    """
    system = platform.system().lower()
    machine = sysconfig.get_platform().split("-")[-1].lower()
    is_64bit = sys.maxsize > 2 ** 32

    if system == "darwin": # get machine architecture of multiarch binaries
        if any([x in machine for x in ("fat", "intel", "universal", "arm64")]):
            machine = platform.machine().lower()
        else:
            machine = 'unknown'

    elif system == "linux":  # fix running 32bit interpreter on 64bit system
        if not is_64bit and machine == "x86_64":
            machine = "i686"
        elif not is_64bit and machine == "aarch64":
                machine = "armv7l"

    elif system == "windows": # return more precise machine architecture names
        if machine == "amd64":
            machine = "x64"
        elif machine == "win32":
            if is_64bit:
                machine = platform.machine().lower()
            else:
                machine = "x86"

    # some more fixes based on examples in https://en.wikipedia.org/wiki/Uname
    if not is_64bit and machine in ("x86_64", "amd64"):
        if any([x in system for x in ("cygwin", "mingw", "msys")]):
            machine = "i686"
        else:
            machine = "i386"

    return f"{system}_{machine}"
