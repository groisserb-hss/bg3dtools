"""ConvergenceScheduler: the windowed relative-change test, its degenerate cases and the max_iter semantics
that affine_ICP, bundle adjustment and nonrigid_ICP rely on."""
import numpy as np

from bg3dtools.utils import ConvergenceScheduler


def _pushed(values, **kw):
    s = ConvergenceScheduler(**kw)
    for v in values:
        s.push(v)
    return s


def test_plateau_at_zero_completes():
    # six zeros: the quotient 0/0 used to read nan, and nan < thresh never held — an endless loop
    assert _pushed([0.0] * 6, thresh=0.05, window=3).complete


def test_plateau_off_zero_completes():
    assert _pushed([2.0] * 6, thresh=0.05, window=3).complete


def test_drop_to_zero_waits_one_window():
    s = _pushed([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], thresh=0.05, window=3)
    assert s.ratio() == np.inf and not s.complete       # fell to zero this window: not settled yet
    for _ in range(3):
        s.push(0.0)
    assert s.complete                                   # the next window is a plateau


def test_ratio_unchanged_off_plateau():
    s = _pushed([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], thresh=0.05, window=3)
    assert s.ratio() == 1.0 and not s.complete


def test_minimum_two_windows_before_any_verdict():
    s = _pushed([1.0] * 5, thresh=np.inf, window=3)
    assert not s.complete
    s.push(1.0)
    assert s.complete


def test_max_iter_is_strict():
    # max_iter=N completes after the (N+1)th push, not the Nth: the semantics affine_ICP and the bundle
    # adjustment were written against, and why nonrigid_ICP bounds its own loop instead
    s = _pushed([5.0, 4.0, 3.0], thresh=0.0, window=1, max_iter=3)
    assert not s.complete
    s.push(2.0)
    assert s.complete
