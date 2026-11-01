"""bg3dtools.image_tools.masks.clean_binary_mask: closing, opening, small components, the hole cap, the border."""
import numpy as np
import pytest

from bg3dtools.image_tools import clean_binary_mask


def test_no_steps_returns_a_copy_and_empty_stays_empty():
    m = np.zeros((6, 6), bool)
    assert not clean_binary_mask(m, close_px=1, open_px=1, min_area=5, max_hole_area=5).any()
    m[1:5, 1:5] = True
    out = clean_binary_mask(m)
    np.testing.assert_array_equal(out, m)
    assert out is not m
    with pytest.raises(ValueError, match="2-D"):
        clean_binary_mask(np.zeros((2, 2, 2), bool))


def test_closing_bridges_a_one_pixel_gap():
    m = np.zeros((7, 12), bool)
    m[2:5, 1:5] = True
    m[2:5, 6:11] = True                                   # column 5 is the gap
    out = clean_binary_mask(m, close_px=1)
    assert out[2:5, 5].all() and out.sum() == m.sum() + 3


def test_opening_removes_specks_and_keeps_bars():
    m = np.zeros((9, 12), bool)
    m[1, 1] = True                                        # a 1-px speck
    m[3:8, 2:10] = True                                   # a 5x8 bar
    out = clean_binary_mask(m, open_px=1)
    assert not out[1, 1] and out[3:8, 2:10].all() and out.sum() == 40


def test_small_components_are_dropped():
    m = np.zeros((10, 10), bool)
    m[1:3, 1:3] = True                                    # 4 px
    m[5:9, 5:9] = True                                    # 16 px
    out = clean_binary_mask(m, min_area=5)
    assert not out[1:3, 1:3].any() and out[5:9, 5:9].all() and out.sum() == 16


def test_hole_cap_fills_small_holes_only():
    m = np.ones((20, 20), bool)
    m[2:4, 2:4] = False                                   # a 4-px hole (a highlight)
    m[8:18, 8:18] = False                                 # a 100-px hole (a slat gap)
    out = clean_binary_mask(m, max_hole_area=10)
    assert out[2:4, 2:4].all() and not out[8:18, 8:18].any()
    assert not clean_binary_mask(m, max_hole_area=0)[2:4, 2:4].any()


def test_regions_touching_the_border_are_not_eroded():
    m = np.zeros((8, 8), bool)
    m[:, :4] = True
    np.testing.assert_array_equal(clean_binary_mask(m, close_px=1, open_px=1), m)
