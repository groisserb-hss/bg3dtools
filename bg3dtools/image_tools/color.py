"""
Colour-space conversions: sRGB -> CIE XYZ -> CIELAB -> LCh(ab).

CIELAB separates lightness ``L*`` from the two opponent axes ``a*`` (green-red) and ``b*`` (blue-yellow); its
polar form ``LCh`` gives an ABSOLUTE chroma ``C*`` and a hue angle ``h``. Unlike HSV -- whose saturation is
relative to the pixel's own brightness, so a near-black pixel's hue is decided by a count or two of JPEG noise
-- a dark colour has a small chroma, and a chroma floor removes it from any hue rule honestly.

Conventions: sRGB in [0, 1] (IEC 61966-2-1 transfer function), D65 white, CIE 1976 L*a*b*. Inputs are
``(..., 3)`` arrays; outputs keep the leading shape.
"""

import numpy as np

__all__ = [
    "D65_WHITE",
    "srgb_to_linear",
    "srgb_to_xyz",
    "xyz_to_lab",
    "srgb_to_lab",
    "lab_to_lch",
    "srgb_to_lch",
    "hue_in_window",
]

# D65 reference white, Y = 1 (CIE 1931 2-degree observer)
D65_WHITE = np.array([0.95047, 1.0, 1.08883])

# Linear sRGB -> XYZ (IEC 61966-2-1, D65)
_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])

_EPS = 216.0 / 24389.0      # CIE's exact (6/29)^3
_KAPPA = 24389.0 / 27.0     # CIE's exact (29/3)^3


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer function: gamma-encoded ``rgb`` in [0, 1] -> linear light."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> CIE XYZ (D65, Y of white = 1)."""
    return srgb_to_linear(rgb) @ _SRGB_TO_XYZ.T


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray = D65_WHITE) -> np.ndarray:
    """CIE XYZ -> CIELAB ``(L*, a*, b*)``, ``L*`` in [0, 100]."""
    t = np.asarray(xyz, dtype=np.float64) / white
    f = np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> CIELAB (D65). Values outside [0, 1] are clipped first."""
    return xyz_to_lab(srgb_to_xyz(np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)))


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    """CIELAB -> ``(L*, C*, h)``: chroma ``C* = |(a*, b*)|`` and hue angle ``h`` in degrees, [0, 360)."""
    lab = np.asarray(lab, dtype=np.float64)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    return np.stack([lab[..., 0], chroma, hue], axis=-1)


def srgb_to_lch(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> ``(L*, C*, h_deg)`` (CIELAB, D65)."""
    return lab_to_lch(srgb_to_lab(rgb))


def hue_in_window(hue_deg: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """True where ``hue_deg`` lies in the closed arc from ``lo`` to ``hi`` going UP (counter-clockwise);
    ``lo > hi`` wraps through 0 (``(340, 20)`` = reds). Angles are taken modulo 360."""
    h = np.asarray(hue_deg, dtype=np.float64) % 360.0
    lo, hi = float(lo) % 360.0, float(hi) % 360.0
    return ((h >= lo) & (h <= hi)) if lo <= hi else ((h >= lo) | (h <= hi))
