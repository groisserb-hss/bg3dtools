"""Shared GPU/CPU delegate selection for the MediaPipe Tasks detectors.

MediaPipe has no CUDA backend: its "GPU" delegate is GL-based (OpenGL ES / GL
compute), so it needs a usable GL context and is absent from some platforms and
pip wheels -- headless Linux without EGL, and the macOS Tasks wheels, are the
known-shaky cases. We therefore *attempt* the requested delegate and fall back to
CPU if the detector cannot be created, logging the delegate actually used.

**GPU is opt-in.** The default is CPU, which is what every caller got before this
module existed. Two reasons to keep it that way: MediaPipe's GL failures are not
always catchable -- an absl ``CHECK`` failure aborts the process natively, and no
``except`` can rescue that -- and GPU and CPU TFLite inference are not
bit-identical, so flipping the default would silently change landmark values
depending on the host. Opt in per call with ``use_gpu=True``, or per process with
``MEDIAPIPE_USE_GPU=1``. The env var propagates to ``multiprocessing`` *spawn*
workers without threading a flag through every call site. (Note GL contexts are
not fork-safe: under the *fork* start method, create detectors in the child.)

Once a detector kind's GPU delegate fails, it is not retried in that process --
otherwise a GPU-less host pays a failed GL init on every single detector
creation, silently, with no log line after the first.
"""
import logging
import os
from typing import Any, Callable, Optional

__all__ = ["gpu_requested", "create_detector"]

log = logging.getLogger("bg3dtools.mediapipe")

# Log the resolved delegate once per (detector kind, delegate) per process: one
# line per camera/scan is noise, but keying on the pair means a later switch is
# still reported rather than hidden behind the first resolution.
_logged = set()

# Detector kinds whose GPU delegate raised once already; see module docstring.
_gpu_unavailable = set()

_TRUTHY = ("1", "true", "yes", "on")


def _delegates():
    """Return the ``(gpu, cpu)`` delegate enum values.

    Isolated into its own function for two reasons: it keeps ``mediapipe`` (the
    ``vision`` extra) lazily imported per repo convention, and it gives tests a
    single seam to stub, so the fallback logic can be exercised without the extra
    installed.
    """
    from mediapipe.tasks.python.core.base_options import BaseOptions
    return BaseOptions.Delegate.GPU, BaseOptions.Delegate.CPU


def _reset_caches():
    """Clear the per-process log and GPU-availability caches. For tests."""
    _logged.clear()
    _gpu_unavailable.clear()


def gpu_requested(use_gpu: Optional[bool] = None) -> bool:
    """Whether to try the GPU delegate.

    Parameters
    ----------
    use_gpu : bool, optional
        Explicit override; wins over the environment. ``None`` (default) defers
        to ``MEDIAPIPE_USE_GPU``.

    Returns
    -------
    bool
        True only if explicitly requested, or if ``MEDIAPIPE_USE_GPU`` is one of
        ``1``/``true``/``yes``/``on`` (case- and whitespace-insensitive).
        Defaults to False -- GPU is opt-in.
    """
    if use_gpu is not None:
        return bool(use_gpu)
    return os.environ.get("MEDIAPIPE_USE_GPU", "0").strip().lower() in _TRUTHY


def _log_once(what: str, delegate_name: str) -> None:
    key = (what, delegate_name)
    if key not in _logged:
        log.info("%s: using %s delegate", what, delegate_name)
        _logged.add(key)


def create_detector(
    build: Callable[[Any], Any],
    use_gpu: Optional[bool] = None,
    what: str = "MediaPipe",
) -> Any:
    """Build a Tasks detector on the GPU delegate if asked, else on CPU.

    Parameters
    ----------
    build : callable
        ``(delegate) -> detector``, where *delegate* is a
        ``BaseOptions.Delegate`` value to thread into ``BaseOptions``.
    use_gpu : bool, optional
        See :func:`gpu_requested`. GPU is opt-in.
    what : str
        Detector name, used for logging and for the per-kind GPU cache.

    Returns
    -------
    object
        The created detector: GPU-backed if requested and constructible, else CPU.

    Notes
    -----
    A GPU failure that surfaces as a Python exception is caught and downgraded.
    One that surfaces as a native ``abort()`` inside MediaPipe's GL layer cannot
    be -- the process dies. That asymmetry is why GPU is opt-in.

    ``build`` runs inside the caller's own ``SuppressCppStderr`` block at two of
    the three call sites, so MediaPipe's stderr diagnosis of a GL failure is
    discarded; only ``type(e).__name__: e`` survives into the warning below.
    """
    gpu, cpu = _delegates()

    if gpu_requested(use_gpu) and what not in _gpu_unavailable:
        try:
            detector = build(gpu)
            _log_once(what, "GPU")
            return detector
        except Exception as e:
            # Wheel/platform without a usable GPU delegate -- degrade, don't die.
            # Remember it, so we don't re-pay a failed GL init on every creation.
            _gpu_unavailable.add(what)
            log.warning("%s: GPU delegate unavailable (%s: %s); falling back to CPU",
                        what, type(e).__name__, e)

    detector = build(cpu)
    _log_once(what, "CPU")
    return detector
