"""bg3dtools.image_tools.color: sRGB -> CIELAB / LCh against published reference values (D65), the hue window, and
the face-centroid lookup of an image in texture space."""
import numpy as np

from bg3dtools.image_tools.color import hue_in_window, lab_to_lch, srgb_to_lab, srgb_to_lch
from bg3dtools.mesh.texture import sample_face_centroids

# sRGB primaries / greys -> CIELAB (D65), the standard reference values (e.g. Lindbloom's calculator)
REFERENCE = {
    (1.0, 1.0, 1.0): (100.0, 0.0, 0.0),
    (0.0, 0.0, 0.0): (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0): (53.2408, 80.0925, 67.2032),
    (0.0, 1.0, 0.0): (87.7347, -86.1827, 83.1793),
    (0.0, 0.0, 1.0): (32.2970, 79.1875, -107.8602),
    (0.5, 0.5, 0.5): (53.3889, 0.0, 0.0),
}


def test_srgb_to_lab_matches_reference():
    rgb = np.array(list(REFERENCE))
    np.testing.assert_allclose(srgb_to_lab(rgb), np.array(list(REFERENCE.values())), atol=2e-3)


def test_shapes_and_clipping():
    img = np.random.default_rng(0).random((4, 5, 3))
    assert srgb_to_lab(img).shape == (4, 5, 3) and srgb_to_lch(img).shape == (4, 5, 3)
    np.testing.assert_allclose(srgb_to_lab([[1.2, -0.1, 0.5]]), srgb_to_lab([[1.0, 0.0, 0.5]]))


def test_float32_stays_float32():
    rgb = np.array(list(REFERENCE), dtype=np.float32)
    lab = srgb_to_lab(rgb)
    assert lab.dtype == np.float32 and srgb_to_lch(rgb).dtype == np.float32
    np.testing.assert_allclose(lab, np.array(list(REFERENCE.values())), atol=1e-2)
    assert srgb_to_lab(np.zeros((2, 3), np.uint8)).dtype == np.float64           # integers are promoted


def test_lch_polar_form():
    lch = lab_to_lch(np.array([[50.0, 0.0, 10.0], [50.0, -10.0, 0.0], [50.0, 0.0, -10.0], [50.0, 3.0, 4.0]]))
    np.testing.assert_allclose(lch[:, 1], [10.0, 10.0, 10.0, 5.0])
    np.testing.assert_allclose(lch[:, 2], [90.0, 180.0, 270.0, np.degrees(np.arctan2(4, 3))])
    # pure sRGB blue: hue angle ~306 deg, the blue sector
    assert abs(srgb_to_lch([[0.0, 0.0, 1.0]])[0, 2] - 306.29) < 0.05


def test_hue_window_wraps():
    h = np.array([0.0, 10.0, 200.0, 345.0, 360.0, 370.0])
    assert hue_in_window(h, 250, 320).tolist() == [False, False, False, False, False, False]
    assert hue_in_window(h, 340, 20).tolist() == [True, True, False, True, True, True]
    assert hue_in_window(h, 190, 210).tolist() == [False, False, True, False, False, False]


def test_sample_face_centroids():
    img = np.zeros((2, 2, 3), np.uint8); img[0, 0] = (255, 0, 0); img[1, 1] = (0, 0, 255)   # row 0 = top (v = 1)
    tc = np.array([[0.25, 0.75], [0.75, 0.25]])
    ftc = np.array([[0, 0, 0], [1, 1, 1]])                       # face 0 in the top-left texel, face 1 bottom-right
    np.testing.assert_array_equal(sample_face_centroids(img, tc, ftc), [[255, 0, 0], [0, 0, 255]])
    # a 2-D map in the same UV space (a mask of the texture at HALF resolution: 1x1) is looked up on its own grid
    mask = np.array([[True]])
    np.testing.assert_array_equal(sample_face_centroids(mask, tc, ftc), [[True], [True]])
    mask4 = np.zeros((4, 4), bool); mask4[:2, :2] = True         # a mask at DOUBLE resolution: only the top-left face
    np.testing.assert_array_equal(sample_face_centroids(mask4, tc, ftc)[:, 0], [True, False])
