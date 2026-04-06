"""
Regression tests for performance optimizations.

Each test verifies that an optimized function produces the same output
as the expected mathematical result or a reference implementation.
"""

import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def icosahedron():
    """Standard icosahedron mesh for testing."""
    import igl
    v, f = igl.read_triangle_mesh(igl.TUTORIAL_SHARED_PATH + "/bunny.off")
    if v.shape[0] == 0:
        pytest.skip("Could not load test mesh")
    return v.astype(np.float64), f.astype(np.int64)


@pytest.fixture
def small_mesh():
    """Simple 4-vertex tetrahedron mesh."""
    v = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3)/2, 0],
        [0.5, np.sqrt(3)/6, np.sqrt(6)/3],
    ], dtype=np.float64)
    f = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3],
    ], dtype=np.int64)
    return v, f


@pytest.fixture
def cube_mesh():
    """Simple cube mesh (8 verts, 12 faces)."""
    v = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=np.float64)
    f = np.array([
        [0,1,2],[0,2,3],  # bottom
        [4,6,5],[4,7,6],  # top
        [0,4,5],[0,5,1],  # front
        [2,6,7],[2,7,3],  # back
        [0,3,7],[0,7,4],  # left
        [1,5,6],[1,6,2],  # right
    ], dtype=np.int64)
    return v, f


# ===========================================================================
# H1: compute.py — .diagonal() instead of .todense()
# ===========================================================================

class TestH1DiagonalExtraction:
    def test_sparse_diagonal(self):
        """Verify sparse .diagonal() gives same result as np.diag(.todense())."""
        row = np.array([0, 1, 2, 3, 0, 1])
        col = np.array([0, 1, 2, 3, 1, 0])
        data = np.array([5.0, 3.0, 7.0, 2.0, 0.5, 0.5])
        A = sparse.csr_matrix((data, (row, col)), shape=(4, 4))
        expected = np.abs(np.diag(A.todense()))
        result = np.abs(A.diagonal())
        np.testing.assert_array_equal(result, expected)


# ===========================================================================
# H2: clean.py — sparse.diags() instead of np.diag()
# ===========================================================================

class TestH2SparseDiags:
    def test_sparse_diags_matches_dense(self):
        """sparse.diags(w) should match csr_matrix(np.diag(w))."""
        w = np.array([1.0, 2.0, 3.0, 4.0])
        expected = sparse.csr_matrix(np.diag(w))
        result = sparse.diags(w, format='csr')
        np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())


# ===========================================================================
# H3: laplace.py — diagonal fast path for spsolve(M, L)
# ===========================================================================

class TestH3DiagonalFastPath:
    def test_laplacian_smoothing(self, small_mesh):
        """laplacian_smoothing should work with diagonal mass matrix."""
        from bg3dtools.mesh.laplace import cotangent_weights, fem_mass_matrix, laplacian_smoothing

        v, f = small_mesh
        L = cotangent_weights(v, f)
        M = fem_mass_matrix(v, f)
        signal = np.random.RandomState(42).rand(v.shape[0], 2)
        result = laplacian_smoothing(L, M, signal)
        assert result.shape == signal.shape
        # Smoothed signal should differ from original
        assert not np.allclose(result, signal)


# ===========================================================================
# H4: generate.py — vectorized cube splatting
# ===========================================================================

class TestH4CubeSplatting:
    def test_pointcloud_to_splatted_mesh(self):
        """Vectorized cube splatting should produce correct geometry."""
        from bg3dtools.mesh.generate import pointcloud_to_splatted_mesh
        pts = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float64)
        v, f = pointcloud_to_splatted_mesh(pts, cube_size=0.5)
        # 2 points × 8 verts each = 16 verts
        assert v.shape == (16, 3)
        # 2 points × 12 faces each = 24 faces
        assert f.shape == (24, 3)
        # All face indices should be valid
        assert np.all(f >= 0) and np.all(f < 16)


# ===========================================================================
# H5: utils.py — vectorized face_2_vertex_map
# ===========================================================================

class TestH5Face2VertexMap:
    def test_rows_sum_to_one(self, cube_mesh):
        """Each referenced vertex row of F2V should sum to 1."""
        from bg3dtools.mesh.utils import face_2_vertex_map
        v, f = cube_mesh
        F2V = face_2_vertex_map(v, f)
        row_sums = np.asarray(F2V.sum(axis=1)).ravel()
        referenced = np.unique(f)
        np.testing.assert_allclose(row_sums[referenced], 1.0, atol=1e-12)

    def test_shape(self, cube_mesh):
        """F2V should be (nV, nF)."""
        from bg3dtools.mesh.utils import face_2_vertex_map
        v, f = cube_mesh
        F2V = face_2_vertex_map(v, f)
        assert F2V.shape == (v.shape[0], f.shape[0])


# ===========================================================================
# H6: utils.py — average_onto_vertices with bincount
# ===========================================================================

class TestH6AverageOntoVertices:
    def test_uniform_face_values(self, cube_mesh):
        """Uniform face values should give uniform vertex values."""
        from bg3dtools.mesh.utils import average_onto_vertices
        v, f = cube_mesh
        fv = np.ones(f.shape[0]) * 3.0
        result = average_onto_vertices(v, f, fv)
        np.testing.assert_allclose(result, 3.0, atol=1e-12)

    def test_multichannel(self, cube_mesh):
        """Multi-channel face values should work correctly."""
        from bg3dtools.mesh.utils import average_onto_vertices
        v, f = cube_mesh
        fv = np.column_stack([np.ones(f.shape[0]), np.ones(f.shape[0]) * 2])
        result = average_onto_vertices(v, f, fv)
        assert result.shape == (v.shape[0], 2)
        np.testing.assert_allclose(result[:, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(result[:, 1], 2.0, atol=1e-12)


# ===========================================================================
# H9: quantize.py — fancy indexing in voxelize
# ===========================================================================

class TestH9Voxelize:
    def test_round_trip(self):
        """voxelize → convert_to_points should recover original points."""
        from bg3dtools.pointclouds.quantize import voxelize, convert_to_points
        pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        shape = (10, 10, 10)
        seg = voxelize(pts, shape)
        assert seg.shape == shape
        assert seg.sum() == 3
        recovered = convert_to_points(seg)
        np.testing.assert_array_equal(np.sort(recovered, axis=0),
                                       np.sort(pts, axis=0))


# ===========================================================================
# H10: fitting.py — weighted line fit via sqrt scaling
# ===========================================================================

class TestH10FitLineTo:
    def test_fit_line_collinear(self):
        """Points along x-axis should yield x-direction."""
        from bg3dtools.pointclouds.fitting import fit_line_to_points
        pts = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]], dtype=np.float64)
        center, direction = fit_line_to_points(pts)
        # Direction should be along x-axis (or its negative)
        assert abs(abs(direction[0, 0]) - 1.0) < 1e-6
        assert abs(direction[0, 1]) < 1e-6
        assert abs(direction[0, 2]) < 1e-6

    def test_fit_line_weighted(self):
        """Weighted fit should still find dominant direction."""
        from bg3dtools.pointclouds.fitting import fit_line_to_points
        rng = np.random.RandomState(42)
        # Points along y-axis with small noise
        pts = np.column_stack([rng.randn(50)*0.01, np.linspace(0, 10, 50), rng.randn(50)*0.01])
        w = np.ones(50)
        w[:10] = 5.0  # weight first 10 points more
        center, direction = fit_line_to_points(pts, w)
        # Should still be along y-axis
        assert abs(abs(direction[0, 1]) - 1.0) < 0.1


# ===========================================================================
# H14: geometric_utilities.py — difference_matrix with cdist
# ===========================================================================

class TestH14DifferenceMatrix:
    def test_norm_mode_matches_cdist(self):
        """difference_matrix with norm=True should match cdist."""
        from spectral_match.tools.geometric_utilities import difference_matrix
        rng = np.random.RandomState(42)
        a = rng.rand(10, 3)
        b = rng.rand(8, 3)
        result = difference_matrix(a, b, norm=True)
        expected = cdist(b, a)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_non_norm_mode(self):
        """difference_matrix with norm=False should return difference tensor."""
        from spectral_match.tools.geometric_utilities import difference_matrix
        a = np.array([[1, 2, 3]], dtype=np.float64)
        b = np.array([[4, 5, 6]], dtype=np.float64)
        result = difference_matrix(a, b, norm=False)
        expected = b[:, np.newaxis, :] - a[np.newaxis, :, :]
        np.testing.assert_array_equal(result, expected)


# ===========================================================================
# H16: stats.py — vectorized ICC bootstrap
# ===========================================================================

class TestH16ICCValue:
    def test_icc_perfect_agreement(self):
        """Perfect agreement should give ICC near 1."""
        from bg3dtools.utils.stats import icc_value
        # 10 subjects, 3 raters, all identical ratings
        data = np.tile(np.arange(10, dtype=np.float64).reshape(-1, 1), (1, 3))
        val, ci = icc_value(data, n_boot=50, return_ci=True)
        assert val > 0.95

    def test_icc_returns_confidence(self):
        """ICC should return (value, (low, high)) tuple with return_ci=True."""
        from bg3dtools.utils.stats import icc_value
        rng = np.random.RandomState(42)
        data = rng.rand(20, 4)
        val, ci = icc_value(data, n_boot=50, return_ci=True)
        assert isinstance(val, float)
        assert len(ci) == 2
        assert ci[0] <= val <= ci[1]

    def test_icc_returns_float(self):
        """ICC should return a scalar float by default."""
        from bg3dtools.utils.stats import icc_value
        rng = np.random.RandomState(42)
        data = rng.rand(10, 3)
        val = icc_value(data, n_boot=50)
        assert isinstance(val, float)


# ===========================================================================
# M1: laplace.py — cotangent weights via dot/cross
# ===========================================================================

class TestM1CotangentWeights:
    def test_symmetry(self, cube_mesh):
        """Cotangent weight matrix should be symmetric."""
        from bg3dtools.mesh.laplace import cotangent_weights
        v, f = cube_mesh
        W = cotangent_weights(v, f).tocsr()
        diff = W - W.T
        assert abs(diff).max() < 1e-10

    def test_row_sum_zero(self, cube_mesh):
        """Rows of cotangent weight Laplacian should sum to ~0."""
        from bg3dtools.mesh.laplace import cotangent_weights
        v, f = cube_mesh
        W = cotangent_weights(v, f).tocsr()
        row_sums = np.abs(np.asarray(W.sum(axis=1)).ravel())
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)


# ===========================================================================
# M3: modify.py — vectorized edge_neighbors
# ===========================================================================

class TestM3EdgeNeighbors:
    def test_edge_neighbors_count(self, cube_mesh):
        """Interior edges of a closed mesh should have exactly 2 neighbors."""
        from bg3dtools.mesh.modify import edge_neighbors
        v, f = cube_mesh
        edges, nbrs = edge_neighbors(f)
        assert edges.shape[1] == 2
        assert nbrs.shape[1] == 2
        # For a closed mesh, each edge is shared by exactly 2 faces (no -1)
        assert np.all(nbrs >= 0), "Closed mesh should have no boundary edges"


# ===========================================================================
# M5: quantize.py — convert_to_points with np.argwhere
# ===========================================================================

class TestM5ConvertToPoints:
    def test_basic_mask(self):
        """convert_to_points should find all nonzero voxels."""
        from bg3dtools.pointclouds.quantize import convert_to_points
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1, 2, 3] = True
        mask[0, 0, 0] = True
        pts = convert_to_points(mask)
        assert pts.shape == (2, 3)
        expected = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float64)
        np.testing.assert_array_equal(np.sort(pts, axis=0), expected)


# ===========================================================================
# M13: geometric_utilities.py — reorder_mesh with inverse permutation
# ===========================================================================

class TestM13ReorderMesh:
    def test_reorder_preserves_geometry(self, cube_mesh):
        """Reordering and un-reordering should recover original mesh."""
        from spectral_match.tools.geometric_utilities import reorder_mesh
        v, f = cube_mesh
        idx = np.array([7, 6, 5, 4, 3, 2, 1, 0])  # reverse
        nv, nf = reorder_mesh(v, f, idx)
        # nv[i] should be v[idx[i]]
        np.testing.assert_array_equal(nv, v[idx])
        # Faces should reference correct vertices
        for fi in range(f.shape[0]):
            for vi in range(3):
                old_vid = f[fi, vi]
                new_vid = nf[fi, vi]
                np.testing.assert_array_equal(nv[new_vid], v[old_vid])


# ===========================================================================
# M14: geometric_utilities.py — mesh_neighbours with sparse adjacency
# ===========================================================================

class TestM14MeshNeighbours:
    def test_valence_groups(self, small_mesh):
        """All vertices of a tetrahedron have valence 3."""
        from spectral_match.tools.geometric_utilities import mesh_neighbours
        v, f = small_mesh
        numbers, nodes_list, neighbours_list = mesh_neighbours(f)
        # Tetrahedron: all 4 verts have valence 3
        assert 3 in numbers
        idx = numbers.index(3)
        assert len(nodes_list[idx]) == 4


# ===========================================================================
# M16: np_helpers.py — truncated_normal with scipy.stats
# ===========================================================================

class TestM16TruncatedNormal:
    def test_within_bounds(self):
        """All samples should be within [avg - spread*var, avg + spread*var]."""
        from bg3dtools.utils.np_helpers import truncated_normal
        avg = np.zeros(1000)
        samples = truncated_normal(avg, var=1.0, spread=2.0)
        assert np.all(samples >= -2.0)
        assert np.all(samples <= 2.0)

    def test_shape(self):
        """Output shape should match avg shape."""
        from bg3dtools.utils.np_helpers import truncated_normal
        avg = np.zeros((5, 3))
        samples = truncated_normal(avg)
        assert samples.shape == (5, 3)


# ===========================================================================
# M17: transforms_unified.py — fast path for transform_points_forward
# ===========================================================================

class TestM17TransformPointsForward:
    def test_identity(self):
        """Identity transform should not change points."""
        from bg3dtools.transforms_unified import transform_points_forward
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        aff = np.eye(4, dtype=np.float64)
        result = transform_points_forward(aff, pts)
        np.testing.assert_allclose(result, pts, atol=1e-12)

    def test_translation(self):
        """Translation should shift all points."""
        from bg3dtools.transforms_unified import transform_points_forward
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        aff = np.eye(4, dtype=np.float64)
        aff[:3, 3] = [10, 20, 30]
        result = transform_points_forward(aff, pts)
        expected = pts + np.array([10, 20, 30])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_rotation(self):
        """90-degree rotation about z-axis."""
        from bg3dtools.transforms_unified import transform_points_forward
        pts = np.array([[1, 0, 0]], dtype=np.float64)
        aff = np.eye(4, dtype=np.float64)
        aff[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        result = transform_points_forward(aff, pts)
        expected = np.array([[0, 1, 0]], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_batch_dims(self):
        """Should handle batch dimensions correctly."""
        from bg3dtools.transforms_unified import transform_points_forward
        aff = np.eye(4, dtype=np.float64).reshape(1, 4, 4).repeat(3, axis=0)
        aff[:, :3, 3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pts = np.zeros((3, 5, 3), dtype=np.float64)
        result = transform_points_forward(aff, pts)
        assert result.shape == (3, 5, 3)
        np.testing.assert_allclose(result[0], np.array([[1, 0, 0]]*5), atol=1e-12)

    def test_homogeneous_input(self):
        """4D points should still work via full matmul path."""
        from bg3dtools.transforms_unified import transform_points_forward
        pts = np.array([[1, 0, 0, 1]], dtype=np.float64)
        aff = np.eye(4, dtype=np.float64)
        aff[:3, 3] = [10, 20, 30]
        result = transform_points_forward(aff, pts)
        expected = np.array([[11, 20, 30, 1]], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ===========================================================================
# M18: transforms_unified.py — transform_points_inverse via inverse_rigid
# ===========================================================================

class TestM18TransformPointsInverse:
    def test_round_trip(self):
        """Forward then inverse should recover original points."""
        from bg3dtools.transforms_unified import (
            transform_points_forward, transform_points_inverse, make_aff
        )
        rng = np.random.RandomState(42)
        twist = rng.randn(3) * 0.5
        trans = rng.randn(3) * 10
        aff = make_aff(twist, trans)
        pts = rng.randn(20, 3)
        fwd = transform_points_forward(aff, pts)
        recovered = transform_points_inverse(aff, fwd)
        np.testing.assert_allclose(recovered, pts, atol=1e-8)


# ===========================================================================
# M19: transforms_unified.py — aff_to_rel_params via inverse_rigid
# ===========================================================================

class TestM19AffToRelParams:
    def test_round_trip(self):
        """rel_params_to_aff → aff_to_rel_params should recover params."""
        from bg3dtools.transforms_unified import (
            rel_params_to_aff, aff_to_rel_params
        )
        trunk = [-1, 0, 1]
        rng = np.random.RandomState(42)
        rel_twist = rng.randn(3, 3) * 0.3
        rel_trans = rng.randn(3, 3) * 5
        abs_affs = rel_params_to_aff(trunk, rel_twist, rel_trans)
        rec_twist, rec_trans = aff_to_rel_params(trunk, abs_affs)
        np.testing.assert_allclose(rec_twist, rel_twist, atol=1e-6)
        np.testing.assert_allclose(rec_trans, rel_trans, atol=1e-6)


# ===========================================================================
# M20: iphone/data_io.py — batch Rotation.from_quat
# ===========================================================================

class TestM20BatchRotation:
    def test_batch_quat_conversion(self):
        """Batch Rotation.from_quat should match per-element conversion."""
        from scipy.spatial.transform import Rotation
        rng = np.random.RandomState(42)
        quats = rng.randn(10, 4)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        # Batch
        batch_R = Rotation.from_quat(quats).as_matrix()
        # Per-element
        single_R = np.stack([Rotation.from_quat(q).as_matrix() for q in quats])
        np.testing.assert_allclose(batch_R, single_R, atol=1e-12)


# ===========================================================================
# M22: render/o3d.py — vectorized mesh_to_wireframe edge extraction
# ===========================================================================

class TestM22MeshToWireframe:
    def test_edge_extraction(self, cube_mesh):
        """Vectorized edge extraction should match set-based extraction."""
        v, f = cube_mesh
        # Reference: set-based extraction
        edges_set = set()
        for face in f:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edges_set.add(edge)
        ref_edges = np.array(sorted(edges_set), dtype=np.int32)

        # Optimized: numpy-based extraction
        e = np.concatenate([f[:, [0,1]], f[:, [1,2]], f[:, [2,0]]], axis=0)
        e = np.sort(e, axis=1)
        opt_edges = np.unique(e, axis=0).astype(np.int32)

        np.testing.assert_array_equal(opt_edges, ref_edges)


# ===========================================================================
# inverse_rigid correctness
# ===========================================================================

class TestInverseRigid:
    def test_inverse_rigid_matches_np_inv(self):
        """inverse_rigid should match np.linalg.inv for rigid transforms."""
        from bg3dtools.transforms_unified import inverse_rigid, make_aff
        rng = np.random.RandomState(42)
        twist = rng.randn(3) * 0.5
        trans = rng.randn(3)
        aff = make_aff(twist, trans)
        fast_inv = inverse_rigid(aff)
        np_inv = np.linalg.inv(aff)
        np.testing.assert_allclose(fast_inv, np_inv, atol=1e-10)

    def test_batch_inverse_rigid(self):
        """Batched inverse_rigid should match element-wise np.linalg.inv."""
        from bg3dtools.transforms_unified import inverse_rigid, make_aff
        rng = np.random.RandomState(42)
        twists = rng.randn(5, 3) * 0.5
        trans = rng.randn(5, 3)
        affs = make_aff(twists, trans)
        fast_inv = inverse_rigid(affs)
        np_inv = np.linalg.inv(affs)
        np.testing.assert_allclose(fast_inv, np_inv, atol=1e-10)


# ===========================================================================
# M2: clean.py — nonmanifold_verts with igl.is_vertex_manifold
# ===========================================================================

class TestM2NonmanifoldVerts:
    def test_closed_manifold_has_no_nmv(self, cube_mesh):
        """Closed manifold mesh should have no non-manifold vertices."""
        from bg3dtools.mesh.clean import nonmanifold_verts
        v, f = cube_mesh
        nmv, V2F = nonmanifold_verts(f.astype(np.int64), nV=v.shape[0])
        assert nmv.size == 0
