"""Shared GPU/CPU delegate selection for the MediaPipe Tasks detectors.

MediaPipe has no CUDA backend: its "GPU" delegate is OpenGL / GL-compute based,
and it is not available on every platform or pip wheel (e.g. headless Linux
without an EGL/GL context). So we *attempt* the requested delegate and fall back
to CPU if the detector cannot be created, logging the delegate actually used.

Selection precedence: an explicit ``use_gpu`` argument wins; otherwise the
``MEDIAPIPE_DISABLE_GPU`` environment variable (``1`` -> CPU) decides; the
default is to try the GPU. Using the env var means the choice propagates to
``multiprocessing`` spawn workers without threading a flag through every call.
"""
import logging
import os

from mediapipe.tasks.python.core.base_options import BaseOptions

log = logging.getLogger("bg3dtools.mediapipe")

# Log the resolved delegate once per detector kind per process, to avoid a line
# per camera/scan while still confirming GPU use in each (spawned) worker.
_logged = set()


def gpu_requested(use_gpu=None):
    """Whether to try the GPU delegate.

    Explicit ``use_gpu`` wins; else ``MEDIAPIPE_DISABLE_GPU`` (``1``/``true`` ->
    CPU); default is GPU-when-available.
    """
    if use_gpu is not None:
        return bool(use_gpu)
    return os.environ.get("MEDIAPIPE_DISABLE_GPU", "0").strip().lower() not in (
        "1", "true", "yes", "on")


def create_detector(build, use_gpu=None, what="MediaPipe"):
    """Build a Tasks detector, trying the GPU delegate then falling back to CPU.

    Args:
        build: callable ``(BaseOptions.Delegate) -> detector``.
        use_gpu: None -> decide from env; True/False -> force.
        what: detector name, for logging.

    Returns:
        The created detector (GPU-backed if available and requested, else CPU).
    """
    if gpu_requested(use_gpu):
        try:
            detector = build(BaseOptions.Delegate.GPU)
            if what not in _logged:
                log.info(f"{what}: using GPU delegate (OpenGL)")
                _logged.add(what)
            return detector
        except Exception as e:
            # Wheel/platform without a usable GPU delegate — degrade, don't die.
            log.warning(f"{what}: GPU delegate unavailable "
                        f"({type(e).__name__}: {e}); falling back to CPU")
            _logged.add(what)
    detector = build(BaseOptions.Delegate.CPU)
    if what not in _logged:
        log.info(f"{what}: using CPU delegate")
        _logged.add(what)
    return detector
