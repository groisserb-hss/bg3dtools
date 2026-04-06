"""
Top-level mesh correspondence computation.

Orchestrates functional map solving, zoom-out refinement, and
product manifold filtering to compute dense vertex correspondences
between two meshes.
"""

# import json
import numpy as np

from scipy.optimize import linear_sum_assignment

from .functional_maps import functional_mapping as fm
from .functional_maps import zoom_out as zo
from .product_manifold_filters import product_manifold_filter as pmf
from .deep_functional_maps.operations import ResidualNet
from ..tools.mesh_class import Mesh
from bg3dtools.mesh.utils import bc2sparse, project_to_bccoord, per_vertex_smoothing
from scipy.spatial.distance import cdist

def normalise(x, axis):
    x = x.copy()
    x -= x.min(axis=axis)
    x /= x.max(axis=axis)
    return x


def compute_correspondence(src, dst, config):
    C = fm.correspondence_matrix_solver(src, dst, k=config["initial_solve_dimension"],
                                      optimise=config["symmetry_optimisation"],
                                      euclidean_weight=config["euclidean_init"])

    C = zo.zoomout_refinement(src, dst)(C)
    P = fm.soft_correspondence(src, dst, C, config["euclidean_init"])
    # P = fm.soft_correspondence(src, dst, C, .5)
    assign = lambda x: linear_sum_assignment(x, maximize=True)
    i, j = assign(P)  # initial assignment

    # dg0 = np.inf
    # dg1 = np.abs(dst.g[j][:, j] - src.g[i][:, i]).mean()
    # while dg1 / dg0 < 0.95:
    i, j = pmf.product_manifold_filter_correspondence(assign, src.g, dst.g, i, j, config["product_manifold_filter"])
        # template_on_query = dst.v[j]
        # template_on_query = per_vertex_smoothing(template_on_query, src.f)
        # D = cdist(dst.v, template_on_query)
        # j = np.argmin(D, axis=0)
        # dg0 = dg1
        # dg1 = np.abs(dst.g[j][:, j] - src.g[i][:, i]).mean()

    return i, j




