"""
Tests for transforms_unified.py — verifying rigid_reg, spherical_to_cartesian,
cartesian_to_spherical, and quaternion conversions.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyR

from bg3dtools.transforms_unified import (
    rigid_reg,
    rigid_reg_robust,
    spherical_to_cartesian,
    cartesian_to_spherical,
    transform_points_forward,
    transform_points_inverse,
    twist_to_R,
    R_to_twist,
    twist_to_quat,
    quat_to_twist,
    quat_to_R,
    R_to_quat,
    make_aff,
    inverse,
    inverse_rigid,
    extract_R,
    extract_twist,
    extract_trans,
    extract_params,
    rel_params_to_aff,
    aff_to_rel_params,
)
ATOL = 1e-10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_rotation():
    """Generate a random rotation matrix via QR decomposition."""
    rng = np.random.default_rng(42)
    H = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(H)
    # Ensure proper rotation (det +1)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


def _random_points(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3))


# ===================================================================
# rigid_reg tests
# ===================================================================


class TestRigidReg:

    def test_known_answer(self):
        """Apply a known R+t and recover it."""
        R = _random_rotation()
        t = np.array([1.0, -2.5, 3.3])
        source = _random_points(60, seed=1)
        dest = (R @ source.T).T + t

        aff = rigid_reg(source, dest)
        np.testing.assert_allclose(aff[:3, :3], R, atol=ATOL)
        np.testing.assert_allclose(aff[:3, 3], t, atol=ATOL)

    def test_known_answer_scale(self):
        """Apply a known R+t+s and recover it."""
        R = _random_rotation()
        t = np.array([0.5, -1.0, 2.0])
        s = 2.7
        source = _random_points(80, seed=2)
        dest = (s * R @ source.T).T + t

        aff = rigid_reg(source, dest, scale=True)
        np.testing.assert_allclose(aff[:3, :3], s * R, atol=ATOL)
        np.testing.assert_allclose(aff[:3, 3], t, atol=ATOL)

    def test_return_aligned(self):
        """Verify return_aligned matches transform_points_forward."""
        R = _random_rotation()
        t = np.array([5.0, -3.0, 1.0])
        source = _random_points(40, seed=3)
        dest = (R @ source.T).T + t

        aff, aligned = rigid_reg(source, dest, return_aligned=True)
        expected = transform_points_forward(aff, source)
        np.testing.assert_allclose(aligned, expected, atol=ATOL)
        np.testing.assert_allclose(aligned, dest, atol=ATOL)

    def test_return_aligned_identity(self):
        """return_aligned on identity case returns copy of source."""
        pts = _random_points(10, seed=99)
        aff, aligned = rigid_reg(pts, pts, return_aligned=True)
        np.testing.assert_allclose(aff, np.eye(4), atol=ATOL)
        np.testing.assert_allclose(aligned, pts, atol=ATOL)

    def test_nan_handling(self):
        """NaN rows in source/dest are filtered before registration."""
        R = _random_rotation()
        t = np.array([1.0, 2.0, 3.0])
        source = _random_points(50, seed=4)
        dest = (R @ source.T).T + t

        # Inject NaN in different rows of source and dest
        source_nan = source.copy()
        dest_nan = dest.copy()
        source_nan[0, :] = np.nan
        source_nan[10, 1] = np.nan
        dest_nan[5, 2] = np.nan

        aff = rigid_reg(source_nan, dest_nan)
        np.testing.assert_allclose(aff[:3, :3], R, atol=1e-9)
        np.testing.assert_allclose(aff[:3, 3], t, atol=1e-9)

    def test_identity(self):
        """When source == dest, return eye(4)."""
        pts = _random_points(20, seed=5)
        aff = rigid_reg(pts, pts)
        np.testing.assert_allclose(aff, np.eye(4), atol=ATOL)

    def test_reflection_case(self):
        """Construct a case where naive SVD would produce det(R) < 0."""
        # Points in a plane => one singular value is zero,
        # making the reflection correction path more likely.
        rng = np.random.default_rng(6)
        source = np.zeros((30, 3))
        source[:, :2] = rng.standard_normal((30, 2))

        # Apply a rotation that flips one axis
        R_flip = np.diag([1.0, 1.0, -1.0])  # det = -1, not a rotation
        # Compose with a proper rotation to test the correction
        R_proper = _random_rotation()
        R = R_proper @ R_flip
        # If det(R) < 0, the algorithm should still produce a proper rotation
        t = np.array([1.0, 0.0, 0.0])
        dest = (R @ source.T).T + t

        aff = rigid_reg(source, dest)
        recovered_R = aff[:3, :3]
        # Must be a proper rotation
        assert np.linalg.det(recovered_R) > 0, "det(R) should be positive"

    def test_roundtrip(self):
        """rigid_reg result applied to source recovers dest."""
        R = _random_rotation()
        t = np.array([2.0, -1.0, 0.5])
        source = _random_points(70, seed=7)
        dest = (R @ source.T).T + t

        aff = rigid_reg(source, dest)
        recovered = transform_points_forward(aff, source)
        np.testing.assert_allclose(recovered, dest, atol=ATOL)


# ===================================================================
# spherical_to_cartesian tests
# ===================================================================


class TestSphericalToCartesian:

    def test_known_x_axis(self):
        """(theta=0, phi=pi/2) -> (1, 0, 0)."""
        result = spherical_to_cartesian(np.array([0.0, np.pi / 2]))
        np.testing.assert_allclose(result, [1, 0, 0], atol=ATOL)

    def test_known_y_axis(self):
        """(theta=pi/2, phi=pi/2) -> (0, 1, 0)."""
        result = spherical_to_cartesian(np.array([np.pi / 2, np.pi / 2]))
        np.testing.assert_allclose(result, [0, 1, 0], atol=ATOL)

    def test_known_z_axis(self):
        """(theta=0, phi=0) -> (0, 0, 1)."""
        result = spherical_to_cartesian(np.array([0.0, 0.0]))
        np.testing.assert_allclose(result, [0, 0, 1], atol=ATOL)

    def test_with_rho(self):
        """(theta=0, phi=pi/2, rho=5) -> (5, 0, 0)."""
        result = spherical_to_cartesian(np.array([0.0, np.pi / 2, 5.0]))
        np.testing.assert_allclose(result, [5, 0, 0], atol=ATOL)



# ===================================================================
# cartesian_to_spherical tests
# ===================================================================


class TestCartesianToSpherical:

    def test_known_x_axis(self):
        """(1, 0, 0) -> (theta=0, phi=pi/2, rho=1)."""
        result = cartesian_to_spherical(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [0, np.pi / 2, 1], atol=ATOL)

    def test_known_z_axis(self):
        """(0, 0, 1) -> (theta=0, phi=0, rho=1)."""
        result = cartesian_to_spherical(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(result, [0, 0, 1], atol=ATOL)

    def test_roundtrip_single(self):
        """spherical_to_cartesian(cartesian_to_spherical(v)) ≈ v."""
        rng = np.random.default_rng(60)
        for _ in range(20):
            v = rng.standard_normal(3)
            sph = cartesian_to_spherical(v)
            recovered = spherical_to_cartesian(sph)
            np.testing.assert_allclose(recovered, v, atol=ATOL)

    def test_vectorized(self):
        """Batch (N,3) input produces (N,3) output."""
        rng = np.random.default_rng(70)
        vecs = rng.standard_normal((15, 3))
        result = cartesian_to_spherical(vecs)
        assert result.shape == (15, 3)
        # Verify each row matches scalar version
        for i in range(15):
            scalar = cartesian_to_spherical(vecs[i])
            np.testing.assert_allclose(result[i], scalar, atol=ATOL)



# ===================================================================
# Torch backend tests
# ===================================================================


class TestRigidRegTorch:

    def test_known_answer(self):
        torch = pytest.importorskip("torch")
        R = _random_rotation()
        t = np.array([1.0, -2.5, 3.3])
        source_np = _random_points(60, seed=1)
        dest_np = (R @ source_np.T).T + t

        source = torch.from_numpy(source_np)
        dest = torch.from_numpy(dest_np)

        aff = rigid_reg(source, dest)
        assert isinstance(aff, torch.Tensor)
        np.testing.assert_allclose(aff.numpy()[:3, :3], R, atol=ATOL)
        np.testing.assert_allclose(aff.numpy()[:3, 3], t, atol=ATOL)

    def test_known_answer_scale(self):
        torch = pytest.importorskip("torch")
        R = _random_rotation()
        t = np.array([0.5, -1.0, 2.0])
        s = 2.7
        source_np = _random_points(80, seed=2)
        dest_np = (s * R @ source_np.T).T + t

        source = torch.from_numpy(source_np)
        dest = torch.from_numpy(dest_np)

        aff = rigid_reg(source, dest, scale=True)
        assert isinstance(aff, torch.Tensor)
        np.testing.assert_allclose(aff.numpy()[:3, :3], s * R, atol=ATOL)
        np.testing.assert_allclose(aff.numpy()[:3, 3], t, atol=ATOL)

    def test_return_aligned(self):
        torch = pytest.importorskip("torch")
        R = _random_rotation()
        t = np.array([5.0, -3.0, 1.0])
        source_np = _random_points(40, seed=3)
        dest_np = (R @ source_np.T).T + t

        source = torch.from_numpy(source_np)
        dest = torch.from_numpy(dest_np)

        aff, aligned = rigid_reg(source, dest, return_aligned=True)
        assert isinstance(aff, torch.Tensor)
        assert isinstance(aligned, torch.Tensor)
        np.testing.assert_allclose(aligned.numpy(), dest_np, atol=ATOL)

    def test_identity(self):
        torch = pytest.importorskip("torch")
        pts_np = _random_points(20, seed=5)
        pts = torch.from_numpy(pts_np)
        aff = rigid_reg(pts, pts)
        assert isinstance(aff, torch.Tensor)
        np.testing.assert_allclose(aff.numpy(), np.eye(4), atol=ATOL)

    def test_parity_with_numpy(self):
        """torch and numpy backends produce same results."""
        torch = pytest.importorskip("torch")
        for seed in range(5):
            source_np = _random_points(100, seed=seed + 100)
            rng = np.random.default_rng(seed + 200)
            R = _random_rotation()
            t = rng.standard_normal(3)
            dest_np = (R @ source_np.T).T + t
            dest_np += rng.standard_normal(dest_np.shape) * 0.01

            aff_np = rigid_reg(source_np, dest_np)
            aff_torch = rigid_reg(
                torch.from_numpy(source_np),
                torch.from_numpy(dest_np),
            )
            np.testing.assert_allclose(
                aff_torch.numpy(), aff_np, atol=1e-9,
                err_msg=f"Mismatch at seed={seed}",
            )


class TestRigidRegWeightedRobust:

    def test_weighted_known_answer(self):
        """Weighted rigid_reg recovers a known R+t (clean data -> any weights agree)."""
        R = _random_rotation(); t = np.array([1.0, -2.0, 0.5])
        source = _random_points(80, seed=11); dest = (R @ source.T).T + t
        w = np.random.default_rng(0).random(80) + 0.1
        aff = rigid_reg(source, dest, weights=w)
        np.testing.assert_allclose(aff[:3, :3], R, atol=ATOL)
        np.testing.assert_allclose(aff[:3, 3], t, atol=ATOL)

    def test_weights_downweight_outlier(self):
        """A point given ~0 weight is ignored -> a gross outlier doesn't bias the fit."""
        R = _random_rotation(); t = np.array([0.0, 1.0, -1.0])
        source = _random_points(60, seed=12); dest = (R @ source.T).T + t
        dest[0] += 100.0                                     # gross outlier
        w = np.ones(60); w[0] = 1e-9
        aff = rigid_reg(source, dest, weights=w)
        np.testing.assert_allclose(aff[:3, :3], R, atol=1e-6)
        np.testing.assert_allclose(aff[:3, 3], t, atol=1e-6)

    def test_robust_rejects_outliers(self):
        """rigid_reg_robust recovers the transform despite 15% gross outliers."""
        rng = np.random.default_rng(3)
        R = _random_rotation(); t = np.array([2.0, -1.0, 0.5])
        source = _random_points(200, seed=13); dest = (R @ source.T).T + t
        dest += rng.standard_normal(dest.shape) * 0.001
        dest[:30] += rng.standard_normal((30, 3)) * 10.0
        aff = rigid_reg_robust(source, dest, iters=3)
        np.testing.assert_allclose(aff[:3, :3], R, atol=1e-2)
        np.testing.assert_allclose(aff[:3, 3], t, atol=1e-2)

    def test_weighted_parity_with_numpy(self):
        """torch and numpy weighted rigid_reg agree."""
        torch = pytest.importorskip("torch")
        source = _random_points(120, seed=14)
        R = _random_rotation(); t = np.array([1.0, 2.0, 3.0])
        dest = (R @ source.T).T + t + np.random.default_rng(1).standard_normal((120, 3)) * 0.01
        w = np.random.default_rng(2).random(120) + 0.1
        aff_np = rigid_reg(source, dest, weights=w)
        aff_t = rigid_reg(torch.from_numpy(source), torch.from_numpy(dest), weights=torch.from_numpy(w))
        np.testing.assert_allclose(aff_t.numpy(), aff_np, atol=1e-9)

    def test_robust_parity_with_numpy(self):
        """torch and numpy rigid_reg_robust agree (median via quantile)."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(5)
        source = _random_points(150, seed=15)
        R = _random_rotation(); t = np.array([0.5, -2.0, 1.0])
        dest = (R @ source.T).T + t; dest[:20] += rng.standard_normal((20, 3)) * 8.0
        aff_np = rigid_reg_robust(source, dest, iters=3)
        aff_t = rigid_reg_robust(torch.from_numpy(source), torch.from_numpy(dest), iters=3)
        np.testing.assert_allclose(aff_t.numpy(), aff_np, atol=1e-9)


class TestSphericalToCartesianTorch:

    def test_known_x_axis(self):
        torch = pytest.importorskip("torch")
        result = spherical_to_cartesian(torch.tensor([0.0, np.pi / 2], dtype=torch.float64))
        assert isinstance(result, torch.Tensor)
        np.testing.assert_allclose(result.numpy(), [1, 0, 0], atol=ATOL)

    def test_known_z_axis(self):
        torch = pytest.importorskip("torch")
        result = spherical_to_cartesian(torch.tensor([0.0, 0.0], dtype=torch.float64))
        assert isinstance(result, torch.Tensor)
        np.testing.assert_allclose(result.numpy(), [0, 0, 1], atol=ATOL)

    def test_with_rho(self):
        torch = pytest.importorskip("torch")
        result = spherical_to_cartesian(torch.tensor([0.0, np.pi / 2, 5.0], dtype=torch.float64))
        assert isinstance(result, torch.Tensor)
        np.testing.assert_allclose(result.numpy(), [5, 0, 0], atol=ATOL)

    def test_parity_with_numpy(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(50)
        for _ in range(20):
            theta = rng.uniform(-np.pi, np.pi)
            phi = rng.uniform(0, np.pi)
            inp_np = np.array([theta, phi])
            inp_torch = torch.from_numpy(inp_np)

            result_np = spherical_to_cartesian(inp_np)
            result_torch = spherical_to_cartesian(inp_torch)
            assert isinstance(result_torch, torch.Tensor)
            np.testing.assert_allclose(result_torch.numpy(), result_np, atol=ATOL)


class TestCartesianToSphericalTorch:

    def test_known_x_axis(self):
        torch = pytest.importorskip("torch")
        result = cartesian_to_spherical(torch.tensor([1.0, 0.0, 0.0]))
        assert isinstance(result, torch.Tensor)
        np.testing.assert_allclose(result.numpy(), [0, np.pi / 2, 1], atol=ATOL)

    def test_roundtrip_single(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(60)
        for _ in range(20):
            v_np = rng.standard_normal(3)
            v = torch.from_numpy(v_np)
            sph = cartesian_to_spherical(v)
            recovered = spherical_to_cartesian(sph)
            assert isinstance(recovered, torch.Tensor)
            np.testing.assert_allclose(recovered.numpy(), v_np, atol=ATOL)

    def test_vectorized(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(70)
        vecs_np = rng.standard_normal((15, 3))
        vecs = torch.from_numpy(vecs_np)
        result = cartesian_to_spherical(vecs)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (15, 3)
        # Compare with numpy version
        result_np = cartesian_to_spherical(vecs_np)
        np.testing.assert_allclose(result.numpy(), result_np, atol=ATOL)

    def test_parity_with_numpy(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(80)
        for _ in range(20):
            v_np = rng.standard_normal(3)
            v_torch = torch.from_numpy(v_np)
            result_np = cartesian_to_spherical(v_np)
            result_torch = cartesian_to_spherical(v_torch)
            assert isinstance(result_torch, torch.Tensor)
            np.testing.assert_allclose(result_torch.numpy(), result_np, atol=ATOL)


# ===================================================================
# Quaternion conversion tests (numpy)
# ===================================================================


def _canonicalize_quat(q):
    """Ensure w >= 0 for comparison."""
    q = np.asarray(q, dtype=np.float64)
    if q[..., 3] < 0:
        q = -q
    return q


class TestQuaternionConversions:

    # --- Known-value tests ---

    def test_identity_quat_to_R(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        R = quat_to_R(q)
        np.testing.assert_allclose(R, np.eye(3), atol=ATOL)

    def test_identity_quat_to_twist(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        tw = quat_to_twist(q)
        np.testing.assert_allclose(tw, np.zeros(3), atol=ATOL)

    def test_identity_R_to_quat(self):
        R = np.eye(3)
        q = R_to_quat(R)
        np.testing.assert_allclose(q, [0, 0, 0, 1], atol=ATOL)

    def test_90deg_z_quat_to_R(self):
        """90 degrees about z: quat = [0, 0, sin(pi/4), cos(pi/4)]."""
        q = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        R = quat_to_R(q)
        expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_90deg_z_quat_to_twist(self):
        q = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        tw = quat_to_twist(q)
        np.testing.assert_allclose(tw, [0, 0, np.pi/2], atol=1e-12)

    def test_180deg_x(self):
        """180 degrees about x: quat = [1, 0, 0, 0]."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quat_to_R(q)
        expected = np.diag([1.0, -1.0, -1.0])
        np.testing.assert_allclose(R, expected, atol=1e-12)

        tw = quat_to_twist(q)
        np.testing.assert_allclose(tw, [np.pi, 0, 0], atol=1e-12)

    def test_180deg_x_R_to_quat(self):
        R = np.diag([1.0, -1.0, -1.0])
        q = R_to_quat(R)
        # w >= 0 canonical, so [1,0,0,0] or close
        np.testing.assert_allclose(np.abs(q), [1, 0, 0, 0], atol=1e-12)

    # --- Round-trip tests (random, seeded) ---

    def test_twist_roundtrip_via_quat(self):
        rng = np.random.default_rng(100)
        for _ in range(50):
            tw = rng.standard_normal(3) * rng.uniform(0.1, np.pi - 0.1)
            tw = tw / np.linalg.norm(tw) * rng.uniform(0.1, np.pi - 0.1)
            q = twist_to_quat(tw)
            tw2 = quat_to_twist(q)
            np.testing.assert_allclose(tw2, tw, atol=1e-10)

    def test_quat_to_R_vs_twist_to_R(self):
        rng = np.random.default_rng(101)
        for _ in range(50):
            tw = rng.standard_normal(3)
            tw = tw / np.linalg.norm(tw) * rng.uniform(0.01, np.pi - 0.01)
            q = twist_to_quat(tw)
            R_from_q = quat_to_R(q)
            R_from_tw = twist_to_R(tw)
            np.testing.assert_allclose(R_from_q, R_from_tw, atol=1e-10)

    def test_R_roundtrip_via_quat(self):
        rng = np.random.default_rng(102)
        for _ in range(50):
            tw = rng.standard_normal(3)
            tw = tw / np.linalg.norm(tw) * rng.uniform(0.01, np.pi - 0.01)
            R = twist_to_R(tw)
            q = R_to_quat(R)
            R2 = quat_to_R(q)
            np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_R_to_quat_to_twist_vs_R_to_twist(self):
        rng = np.random.default_rng(103)
        for _ in range(50):
            tw = rng.standard_normal(3)
            tw = tw / np.linalg.norm(tw) * rng.uniform(0.01, np.pi - 0.01)
            R = twist_to_R(tw)
            tw_via_q = quat_to_twist(R_to_quat(R))
            tw_direct = R_to_twist(R)
            np.testing.assert_allclose(tw_via_q, tw_direct, atol=1e-10)

    # --- Edge cases ---

    def test_small_angle_roundtrip(self):
        tw = np.array([1e-6, 0.0, 0.0])
        q = twist_to_quat(tw)
        tw2 = quat_to_twist(q)
        np.testing.assert_allclose(tw2, tw, atol=1e-12)

    def test_near_pi_roundtrip(self):
        theta = np.pi - 1e-6
        tw = np.array([theta, 0.0, 0.0])
        q = twist_to_quat(tw)
        R = quat_to_R(q)
        q2 = R_to_quat(R)
        R2 = quat_to_R(q2)
        np.testing.assert_allclose(R2, R, atol=1e-5)

    def test_antipodal_quat(self):
        """q and -q produce the same twist."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        tw1 = quat_to_twist(q)
        tw2 = quat_to_twist(-q)
        np.testing.assert_allclose(tw1, tw2, atol=1e-12)

    def test_non_unit_input(self):
        """Non-unit quaternion is normalized internally."""
        q = np.array([2.0, 0.0, 0.0, 2.0])
        q_norm = q / np.linalg.norm(q)
        R1 = quat_to_R(q)
        R2 = quat_to_R(q_norm)
        np.testing.assert_allclose(R1, R2, atol=1e-12)

    # --- Batch shape tests ---

    def test_quat_to_R_shapes(self):
        for shape in [(4,), (10, 4), (3, 5, 4)]:
            q = np.tile([0, 0, 0, 1.0], shape[:-1] + (1,))
            R = quat_to_R(q)
            assert R.shape == shape[:-1] + (3, 3), f"Failed for input shape {shape}"

    def test_quat_to_twist_shapes(self):
        for shape in [(4,), (10, 4), (3, 5, 4)]:
            q = np.tile([0, 0, 0, 1.0], shape[:-1] + (1,))
            tw = quat_to_twist(q)
            assert tw.shape == shape[:-1] + (3,), f"Failed for input shape {shape}"

    def test_R_to_quat_shapes(self):
        for shape in [(3, 3), (10, 3, 3), (3, 5, 3, 3)]:
            R = np.tile(np.eye(3), shape[:-2] + (1, 1))
            q = R_to_quat(R)
            assert q.shape == shape[:-2] + (4,), f"Failed for input shape {shape}"

    # --- Cross-validation with scipy ---

    def test_quat_to_R_vs_scipy(self):
        rng = np.random.default_rng(200)
        for _ in range(50):
            q = rng.standard_normal(4)
            q = q / np.linalg.norm(q)
            if q[3] < 0:
                q = -q

            R_ours = quat_to_R(q)
            R_scipy = ScipyR.from_quat(q).as_matrix()
            np.testing.assert_allclose(R_ours, R_scipy, atol=1e-12)

    def test_R_to_quat_vs_scipy(self):
        rng = np.random.default_rng(201)
        for _ in range(50):
            tw = rng.standard_normal(3)
            tw = tw / np.linalg.norm(tw) * rng.uniform(0.01, np.pi - 0.01)
            R = twist_to_R(tw)

            q_ours = R_to_quat(R)
            q_scipy = ScipyR.from_matrix(R).as_quat()
            # Canonicalize scipy result
            if q_scipy[3] < 0:
                q_scipy = -q_scipy

            np.testing.assert_allclose(q_ours, q_scipy, atol=1e-12)


# ===================================================================
# Quaternion conversion tests (torch)
# ===================================================================


class TestQuaternionConversionsTorch:

    def test_identity_quat_to_R(self):
        torch = pytest.importorskip("torch")
        q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        R = quat_to_R(q)
        assert isinstance(R, torch.Tensor)
        np.testing.assert_allclose(R.numpy(), np.eye(3), atol=ATOL)

    def test_identity_quat_to_twist(self):
        torch = pytest.importorskip("torch")
        q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        tw = quat_to_twist(q)
        assert isinstance(tw, torch.Tensor)
        np.testing.assert_allclose(tw.numpy(), np.zeros(3), atol=ATOL)

    def test_identity_R_to_quat(self):
        torch = pytest.importorskip("torch")
        R = torch.eye(3, dtype=torch.float64)
        q = R_to_quat(R)
        assert isinstance(q, torch.Tensor)
        np.testing.assert_allclose(q.numpy(), [0, 0, 0, 1], atol=ATOL)

    def test_90deg_z_quat_to_R(self):
        torch = pytest.importorskip("torch")
        q = torch.tensor([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)], dtype=torch.float64)
        R = quat_to_R(q)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R.numpy(), expected, atol=1e-12)

    def test_180deg_x(self):
        torch = pytest.importorskip("torch")
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        R = quat_to_R(q)
        expected = np.diag([1.0, -1.0, -1.0])
        np.testing.assert_allclose(R.numpy(), expected, atol=1e-12)

    def test_roundtrip_twist_via_quat(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(300)
        for _ in range(30):
            tw_np = rng.standard_normal(3)
            tw_np = tw_np / np.linalg.norm(tw_np) * rng.uniform(0.1, np.pi - 0.1)
            tw = torch.from_numpy(tw_np)
            q = twist_to_quat(tw)
            tw2 = quat_to_twist(q)
            np.testing.assert_allclose(tw2.numpy(), tw_np, atol=1e-10)

    def test_roundtrip_R_via_quat(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(301)
        for _ in range(30):
            tw_np = rng.standard_normal(3)
            tw_np = tw_np / np.linalg.norm(tw_np) * rng.uniform(0.01, np.pi - 0.01)
            R = twist_to_R(torch.from_numpy(tw_np))
            q = R_to_quat(R)
            R2 = quat_to_R(q)
            np.testing.assert_allclose(R2.numpy(), R.numpy(), atol=1e-10)

    def test_batch_shapes(self):
        torch = pytest.importorskip("torch")
        q = torch.zeros(3, 5, 4, dtype=torch.float64)
        q[..., 3] = 1.0
        R = quat_to_R(q)
        assert R.shape == (3, 5, 3, 3)
        tw = quat_to_twist(q)
        assert tw.shape == (3, 5, 3)

        R_in = torch.eye(3, dtype=torch.float64).expand(3, 5, 3, 3).contiguous()
        q_out = R_to_quat(R_in)
        assert q_out.shape == (3, 5, 4)

    def test_parity_with_numpy(self):
        """Each function produces matching results for numpy and torch inputs."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(302)
        for _ in range(30):
            # quat_to_R
            q_np = rng.standard_normal(4)
            q_np = q_np / np.linalg.norm(q_np)
            q_t = torch.from_numpy(q_np)
            np.testing.assert_allclose(
                quat_to_R(q_t).numpy(), quat_to_R(q_np), atol=1e-12)

            # quat_to_twist
            np.testing.assert_allclose(
                quat_to_twist(q_t).numpy(), quat_to_twist(q_np), atol=1e-12)

            # R_to_quat
            tw_np = rng.standard_normal(3)
            tw_np = tw_np / np.linalg.norm(tw_np) * rng.uniform(0.01, np.pi - 0.01)
            R_np = twist_to_R(tw_np)
            R_t = torch.from_numpy(R_np)
            np.testing.assert_allclose(
                R_to_quat(R_t).numpy(), R_to_quat(R_np), atol=1e-12)


# ===================================================================
# Twist / Rotation torch tests
# ===================================================================


class TestTwistRotationTorch:

    def test_twist_to_R_known(self):
        """90-degree rotation about z-axis."""
        torch = pytest.importorskip("torch")
        twist = torch.tensor([0.0, 0.0, np.pi / 2], dtype=torch.float64)
        R = twist_to_R(twist)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        assert isinstance(R, torch.Tensor)
        np.testing.assert_allclose(R.numpy(), expected, atol=1e-12)

    def test_twist_to_R_roundtrip(self):
        """twist -> R -> twist recovers original (30 random, angles in [0.1, pi-0.1])."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(400)
        for _ in range(30):
            tw_np = rng.standard_normal(3)
            angle = rng.uniform(0.1, np.pi - 0.1)
            tw_np = tw_np / np.linalg.norm(tw_np) * angle
            tw = torch.from_numpy(tw_np)
            R = twist_to_R(tw)
            tw_rec = R_to_twist(R)
            np.testing.assert_allclose(tw_rec.numpy(), tw_np, atol=1e-6)

    def test_twist_to_R_batch_shapes(self):
        """Verify shapes for (3,), (10,3), (3,5,3) inputs."""
        torch = pytest.importorskip("torch")
        for shape in [(3,), (10, 3), (3, 5, 3)]:
            tw = torch.zeros(shape, dtype=torch.float64)
            R = twist_to_R(tw)
            assert R.shape == shape[:-1] + (3, 3), f"Failed for input shape {shape}"

    def test_twist_to_R_parity_with_numpy(self):
        """Compare torch vs numpy results (30 random)."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(401)
        for _ in range(30):
            tw_np = rng.standard_normal(3) * rng.uniform(0.01, np.pi - 0.01)
            tw_np = tw_np / np.linalg.norm(tw_np) * rng.uniform(0.01, np.pi - 0.01)
            R_np = twist_to_R(tw_np)
            R_t = twist_to_R(torch.from_numpy(tw_np))
            np.testing.assert_allclose(R_t.numpy(), R_np, atol=1e-12)

    def test_R_to_twist_parity_with_numpy(self):
        """Compare torch vs numpy R_to_twist results."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(402)
        for _ in range(30):
            tw_np = rng.standard_normal(3)
            tw_np = tw_np / np.linalg.norm(tw_np) * rng.uniform(0.1, np.pi - 0.1)
            R_np = twist_to_R(tw_np)
            tw_from_np = R_to_twist(R_np)
            tw_from_t = R_to_twist(torch.from_numpy(R_np))
            np.testing.assert_allclose(tw_from_t.numpy(), tw_from_np, atol=1e-10)


# ===================================================================
# Affine ops torch tests
# ===================================================================


class TestAffineOpsTorch:

    def test_make_aff_known(self):
        """Known twist+trans -> verify R and t blocks."""
        torch = pytest.importorskip("torch")
        tw = torch.tensor([0.0, 0.0, np.pi / 2], dtype=torch.float64)
        tr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        aff = make_aff(tw, tr)
        assert isinstance(aff, torch.Tensor)
        assert aff.shape == (4, 4)
        expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(aff.numpy()[:3, :3], expected_R, atol=1e-12)
        np.testing.assert_allclose(aff.numpy()[:3, 3], [1, 2, 3], atol=1e-12)
        np.testing.assert_allclose(aff.numpy()[3], [0, 0, 0, 1], atol=1e-12)

    def test_make_aff_none_inputs(self):
        """twist=None and trans=None cases."""
        torch = pytest.importorskip("torch")
        tw = torch.tensor([0.0, 0.0, np.pi / 4], dtype=torch.float64)
        tr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        # twist=None -> identity rotation
        aff_no_twist = make_aff(None, tr)
        np.testing.assert_allclose(aff_no_twist.numpy()[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(aff_no_twist.numpy()[:3, 3], [1, 2, 3], atol=1e-12)

        # trans=None -> zero translation
        aff_no_trans = make_aff(tw, None)
        np.testing.assert_allclose(aff_no_trans.numpy()[:3, 3], [0, 0, 0], atol=1e-12)

    def test_inverse_roundtrip(self):
        """aff @ inverse(aff) ≈ I."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(410)
        tw = rng.standard_normal(3) * 0.5
        tr = rng.standard_normal(3)
        aff = make_aff(torch.from_numpy(tw), torch.from_numpy(tr))
        aff_inv = inverse(aff)
        product = aff @ aff_inv
        np.testing.assert_allclose(product.numpy(), np.eye(4), atol=1e-10)

    def test_inverse_rigid_roundtrip(self):
        """aff @ inverse_rigid(aff) ≈ I."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(411)
        tw = rng.standard_normal(3) * 0.5
        tr = rng.standard_normal(3)
        aff = make_aff(torch.from_numpy(tw), torch.from_numpy(tr))
        aff_inv = inverse_rigid(aff)
        product = aff @ aff_inv
        np.testing.assert_allclose(product.numpy(), np.eye(4), atol=1e-10)

    def test_inverse_rigid_matches_inverse(self):
        """inverse_rigid ≈ inverse for rigid transforms."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(412)
        tw = rng.standard_normal(3) * 0.5
        tr = rng.standard_normal(3)
        aff = make_aff(torch.from_numpy(tw), torch.from_numpy(tr))
        np.testing.assert_allclose(
            inverse_rigid(aff).numpy(), inverse(aff).numpy(), atol=1e-10)

    def test_extract_params_roundtrip(self):
        """make_aff -> extract_params recovers inputs."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(413)
        tw_np = rng.standard_normal(3)
        tw_np = tw_np / np.linalg.norm(tw_np) * rng.uniform(0.1, np.pi - 0.1)
        tr_np = rng.standard_normal(3)
        tw = torch.from_numpy(tw_np)
        tr = torch.from_numpy(tr_np)
        aff = make_aff(tw, tr)
        tw_rec, tr_rec = extract_params(aff)
        np.testing.assert_allclose(tw_rec.numpy(), tw_np, atol=1e-6)
        np.testing.assert_allclose(tr_rec.numpy(), tr_np, atol=1e-12)

    def test_extract_R(self):
        """extract_R returns the 3x3 block."""
        torch = pytest.importorskip("torch")
        tw = torch.tensor([0.0, 0.0, np.pi / 2], dtype=torch.float64)
        tr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        aff = make_aff(tw, tr)
        R = extract_R(aff)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R.numpy(), expected, atol=1e-12)

    def test_transform_points_forward(self):
        """Known transform on known points."""
        torch = pytest.importorskip("torch")
        tw = torch.tensor([0.0, 0.0, np.pi / 2], dtype=torch.float64)
        tr = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        aff = make_aff(tw, tr)
        pts = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        result = transform_points_forward(aff, pts)
        # 90deg about z rotates (1,0,0) -> (0,1,0), then translate +1 in x
        np.testing.assert_allclose(result.numpy(), [[1.0, 1.0, 0.0]], atol=1e-12)

    def test_transform_points_inverse(self):
        """Forward then inverse recovers original."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(414)
        tw = torch.from_numpy(rng.standard_normal(3) * 0.5)
        tr = torch.from_numpy(rng.standard_normal(3))
        aff = make_aff(tw, tr)
        pts = torch.from_numpy(rng.standard_normal((20, 3)))
        fwd = transform_points_forward(aff, pts)
        rec = transform_points_inverse(aff, fwd)
        np.testing.assert_allclose(rec.numpy(), pts.numpy(), atol=1e-10)

    def test_transform_points_homogeneous(self):
        """4D homogeneous input."""
        torch = pytest.importorskip("torch")
        tw = torch.tensor([0.0, 0.0, np.pi / 2], dtype=torch.float64)
        tr = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        aff = make_aff(tw, tr)
        pts = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        result = transform_points_forward(aff, pts)
        assert result.shape == (1, 4)
        np.testing.assert_allclose(result.numpy()[:, :3], [[1.0, 1.0, 0.0]], atol=1e-12)

    def test_parity_with_numpy(self):
        """All affine functions compared against numpy (30 random)."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(415)
        for _ in range(30):
            tw_np = rng.standard_normal(3)
            tw_np = tw_np / np.linalg.norm(tw_np) * rng.uniform(0.1, np.pi - 0.1)
            tr_np = rng.standard_normal(3)

            tw_t = torch.from_numpy(tw_np)
            tr_t = torch.from_numpy(tr_np)

            aff_np = make_aff(tw_np, tr_np)
            aff_t = make_aff(tw_t, tr_t)
            np.testing.assert_allclose(aff_t.numpy(), aff_np, atol=1e-12)

            np.testing.assert_allclose(
                inverse(aff_t).numpy(), inverse(aff_np), atol=1e-10)
            np.testing.assert_allclose(
                inverse_rigid(aff_t).numpy(), inverse_rigid(aff_np), atol=1e-10)
            np.testing.assert_allclose(
                extract_R(aff_t).numpy(), extract_R(aff_np), atol=1e-12)

            tw_rec_t, tr_rec_t = extract_params(aff_t)
            tw_rec_np, tr_rec_np = extract_params(aff_np)
            np.testing.assert_allclose(tw_rec_t.numpy(), tw_rec_np, atol=1e-10)
            np.testing.assert_allclose(tr_rec_t.numpy(), tr_rec_np, atol=1e-12)

            pts_np = rng.standard_normal((5, 3))
            pts_t = torch.from_numpy(pts_np)
            np.testing.assert_allclose(
                transform_points_forward(aff_t, pts_t).numpy(),
                transform_points_forward(aff_np, pts_np),
                atol=1e-10)


# ===================================================================
# Kinematic chain torch tests
# ===================================================================


class TestKinematicChainTorch:

    def test_rel_to_abs_single_joint(self):
        """1-joint chain (root only)."""
        torch = pytest.importorskip("torch")
        trunk = [-1]
        tw = torch.randn(1, 3, dtype=torch.float64) * 0.3
        tr = torch.randn(1, 3, dtype=torch.float64)
        # Shape (1, 1, 3) — batch=1, joints=1
        tw = tw.unsqueeze(0)
        tr = tr.unsqueeze(0)
        abs_aff = rel_params_to_aff(trunk, tw, tr)
        assert abs_aff.shape == (1, 1, 4, 4)
        # Should equal make_aff of the single joint
        expected = make_aff(tw[:, 0], tr[:, 0])
        np.testing.assert_allclose(abs_aff[:, 0].numpy(), expected.numpy(), atol=1e-10)

    def test_rel_abs_roundtrip(self):
        """rel -> abs -> rel (B=8, J=17 linear chain)."""
        torch = pytest.importorskip("torch")
        torch.manual_seed(500)
        B, J = 8, 17
        trunk = [-1] + list(range(J - 1))

        axis = torch.randn(B, J, 3, dtype=torch.float64)
        axis = axis / axis.norm(dim=-1, keepdim=True)
        angle = torch.rand(B, J, 1, dtype=torch.float64) * (np.pi - 0.2) + 0.1
        tw = axis * angle
        tr = torch.randn(B, J, 3, dtype=torch.float64)

        abs_aff = rel_params_to_aff(trunk, tw, tr)
        tw_rec, tr_rec = aff_to_rel_params(trunk, abs_aff)
        np.testing.assert_allclose(tw_rec.numpy(), tw.numpy(), atol=0.05)
        np.testing.assert_allclose(tr_rec.numpy(), tr.numpy(), atol=1e-4)

    def test_rel_abs_roundtrip_branching(self):
        """Non-linear kinematic tree."""
        torch = pytest.importorskip("torch")
        torch.manual_seed(501)
        # Tree: 0 is root, 1->0, 2->0, 3->1, 4->1, 5->2
        trunk = [-1, 0, 0, 1, 1, 2]
        J = len(trunk)
        B = 4

        axis = torch.randn(B, J, 3, dtype=torch.float64)
        axis = axis / axis.norm(dim=-1, keepdim=True)
        angle = torch.rand(B, J, 1, dtype=torch.float64) * (np.pi - 0.2) + 0.1
        tw = axis * angle
        tr = torch.randn(B, J, 3, dtype=torch.float64)

        abs_aff = rel_params_to_aff(trunk, tw, tr)
        tw_rec, tr_rec = aff_to_rel_params(trunk, abs_aff)
        np.testing.assert_allclose(tw_rec.numpy(), tw.numpy(), atol=0.05)
        np.testing.assert_allclose(tr_rec.numpy(), tr.numpy(), atol=1e-4)

    def test_batch_shapes(self):
        """(J,3), (B,J,3), (2,B,J,3) inputs."""
        torch = pytest.importorskip("torch")
        trunk = [-1, 0, 1]
        J = 3
        for batch_shape in [(), (4,), (2, 4)]:
            tw = torch.zeros(batch_shape + (J, 3), dtype=torch.float64)
            tr = torch.zeros(batch_shape + (J, 3), dtype=torch.float64)
            abs_aff = rel_params_to_aff(trunk, tw, tr)
            assert abs_aff.shape == batch_shape + (J, 4, 4), \
                f"Failed for batch_shape {batch_shape}"

    def test_parity_with_numpy(self):
        """Compare torch vs numpy (5 seeds)."""
        torch = pytest.importorskip("torch")
        trunk = [-1, 0, 0, 1, 1, 2]
        J = len(trunk)
        for seed in range(5):
            rng = np.random.default_rng(600 + seed)
            B = 3
            tw_np = rng.standard_normal((B, J, 3)) * 0.5
            tr_np = rng.standard_normal((B, J, 3))

            abs_np = rel_params_to_aff(trunk, tw_np, tr_np)
            abs_t = rel_params_to_aff(
                trunk,
                torch.from_numpy(tw_np),
                torch.from_numpy(tr_np),
            )
            np.testing.assert_allclose(abs_t.numpy(), abs_np, atol=1e-10,
                                       err_msg=f"rel_params_to_aff mismatch at seed={seed}")

            tw_rec_np, tr_rec_np = aff_to_rel_params(trunk, abs_np)
            tw_rec_t, tr_rec_t = aff_to_rel_params(trunk, abs_t)
            np.testing.assert_allclose(tw_rec_t.numpy(), tw_rec_np, atol=1e-10,
                                       err_msg=f"aff_to_rel_params twist mismatch at seed={seed}")
            np.testing.assert_allclose(tr_rec_t.numpy(), tr_rec_np, atol=1e-10,
                                       err_msg=f"aff_to_rel_params trans mismatch at seed={seed}")
