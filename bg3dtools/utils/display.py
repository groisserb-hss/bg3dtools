"""
Display and GUI detection utilities.

This module provides functions for detecting the display environment
and matplotlib backend configuration.
"""


def matplotlib_is_headless() -> bool:
    """
    Check if matplotlib is running in a headless (non-GUI) environment.

    Returns
    -------
    is_headless : bool
        True if using an offscreen backend (agg, pdf, svg, etc.).
        False for GUI backends, PyCharm SciView, and Jupyter notebooks.
    """
    try:
        import matplotlib
    except ImportError:
        return True

    b = matplotlib.get_backend().lower()

    # Offscreen / non-interactive renderers
    non_gui = {"agg", "pdf", "ps", "svg", "cairo", "template"}
    if b in non_gui:
        return True

    # Standard GUI backends
    if b in {"macosx", "tkagg", "qt5agg", "qtagg", "wxagg", "gtk3agg", "gtk4agg"}:
        return False

    # Module backends
    if b.startswith("module://"):
        # PyCharm SciView
        if b.endswith("backend_interagg"):
            return False
        # Jupyter inline / ipympl / webagg variants show somewhere (not headless)
        if ("inline" in b) or ("nbagg" in b) or ("webagg" in b):
            return False
        # Unknown module backend: assume usable unless proven otherwise
        return False

    # Fallback: assume usable
    return False