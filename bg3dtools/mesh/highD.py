"""
High-dimensional mesh projection utilities.

This module provides classes for projecting points onto mesh surfaces
in arbitrary dimensions using nearest-neighbor acceleration structures.
"""

from typing import Tuple
import igl
import numpy as np
from pykdtree.kdtree import KDTree

__all__ = [
    "MeshProjector",
]


class MeshProjector:
    """
    Fast projection of points onto a triangulated mesh surface.

    Uses KD-trees on vertices, edge midpoints, and face centroids
    to accelerate nearest-simplex queries for high-dimensional meshes.

    Parameters
    ----------
    v : (N, D) ndarray
        Vertex coordinates in D dimensions.
    f : (M, 3) ndarray
        Triangle face indices.

    Attributes
    ----------
    v : ndarray
        Vertex coordinates.
    f : ndarray
        Face indices.
    """

    def __init__(self, v: np.ndarray, f: np.ndarray):
        self.v = v
        self.f = f

        self.vert_tree = KDTree(v)
        v2f, ni = igl.vertex_triangle_adjacency(f, v.shape[0])
        v2f_list = [v2f[ni[vv]:ni[vv + 1]] for vv in range(v.shape[0])]
        self.v2f = v2f_list

        edges = igl.edges(f)
        edge_centers = (v[edges[:, 0], :] + v[edges[:, 1], :]) / 2
        self.edge_tree = KDTree(edge_centers)
        e2f_list = [[]] * edges.shape[0]
        for ee, edge in enumerate(edges):
            e0_f = v2f_list[edge[0]]
            e1_f = v2f_list[edge[1]]
            e2f_list[ee] = np.intersect1d(e0_f, e1_f)
        self.e2f = e2f_list

        face_pts = (v[f[:, 0], :] + v[f[:, 1], :] + v[f[:, 2], :]) / 3
        self.face_tree = KDTree(face_pts)

    def project(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project points onto the mesh surface.

        Parameters
        ----------
        points : (K, D) ndarray
            Query points to project.

        Returns
        -------
        d2 : (K,) ndarray
            Squared distance from each point to its projection.
        c : (K, D) ndarray
            Projected point coordinates on the mesh surface.
        bc : (K, 3) ndarray
            Barycentric coordinates of projections within their faces.
        """
        n = points.shape[0]

        vd, vidx = self.vert_tree.query(points, k=3)
        ed, eidx = self.edge_tree.query(points, k=3)
        fd, fidx = self.face_tree.query(points, k=5)

        # output is squared distance, projected point, and barycentric coordinates for each query point
        d2 = -np.ones(n, dtype=np.float32)
        c = np.zeros_like(points)
        bc = -np.ones_like(points)

        for pp, point in enumerate(points):
            best_d2 = np.inf

            # faces attached to nearest vertices
            v0, v1, v2 = vidx[pp]
            v_faces = np.concatenate((self.v2f[v0], self.v2f[v1], self.v2f[v2]))

            # faces attached to nearest edge centers
            e0, e1, e2 = eidx[pp]
            e_faces = np.concatenate((self.e2f[e0], self.e2f[e1], self.e2f[e2]))

            # faces of nearest barycenters
            f_faces = fidx[pp]

            # search space of potential nearest simplexes
            idx = np.unique(np.concatenate((v_faces, e_faces, f_faces)))
            for i in idx:
                test_d2, test_c, test_bc = igl.point_simplex_squared_distance(point, self.v, self.f, i)

                if test_d2 < best_d2:
                    best_d2 = test_d2
                    d2[pp] = test_d2
                    c[pp] = test_c
                    bc[pp] = test_bc

        return d2, c, bc
