"""
Graph operations for skeleton analysis.

This module provides functions for converting 3D skeletons to graphs
and extracting paths and loops from graph structures.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import scipy.spatial as spatial
import igraph as ig

from bg3dtools.transforms_unified import transform_points_inverse
from bg3dtools.pointclouds.quantize import convert_to_points


def skeleton_to_graph(
    skeleton: np.ndarray,
    affine: np.ndarray
) -> ig.Graph:
    """
    Convert a binary 3D skeleton to a graph.

    Creates a graph where skeleton voxels become vertices and
    neighboring voxels (within sqrt(3) distance) are connected by edges.

    Parameters
    ----------
    skeleton : (X, Y, Z) ndarray
        Binary 3D skeleton volume.
    affine : (4, 4) ndarray
        Affine transformation matrix for coordinate conversion.

    Returns
    -------
    graph : igraph.Graph
        Largest connected component with 'coord' vertex attribute
        containing 3D coordinates.
    """
    # convert skeleton to graph
    log = logging.getLogger('Vertebra.skeleton_to_graph')
    # log.info('convert skeleton to graph')

    pts_scaled = convert_to_points(skeleton, affine)
    pts_homo = transform_points_inverse(affine, pts_scaled)

    tree = spatial.cKDTree(pts_homo, leafsize=8)
    neighbors = tree.query_ball_point(pts_homo, r=1.75, p=2)  # find all points within 1 step of each other
    edges = []
    for n0, neighborhood in enumerate(neighbors):
        neighborhood.remove(n0)
        pairs = [[n0, n] for n in neighborhood]
        edges += [tuple(sorted(p)) for p in pairs]
    edges = list(set(edges))

    graph = ig.Graph(len(pts_scaled), edges)
    # keep track of coordinates
    graph.vs.set_attribute_values('coord', [p for p in pts_scaled])

    # extract largest connected segment
    subgraphs = graph.decompose(minelements=50)
    gg = np.argmax([len(g.vs) for g in subgraphs])

    return subgraphs[gg]


def get_largest_loop(graph: ig.Graph) -> np.ndarray:
    """
    Find the largest loop in a graph.

    Iteratively removes each edge and finds the shortest path between
    its endpoints. The longest such path forms the largest loop.

    Parameters
    ----------
    graph : igraph.Graph
        Input graph with 'coord' vertex attribute.

    Returns
    -------
    loop : (N, 3) ndarray
        3D coordinates of vertices forming the largest loop.
    """

    # extract point cloud
    coords = graph.vs.get_attribute_values('coord')
    coords = np.row_stack(coords)

    # cut
    edges = graph.get_edgelist()
    max_len, long_path = 0, []
    for edge in edges:
        # cut edge
        graph.delete_edges(edge)
        # find shortest path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = graph.get_shortest_path(edge[0], edge[1])
        if len(path) > max_len:
            max_len = len(path)
            long_path = path
        graph.add_edge(edge[0], edge[1])

    loop = coords[long_path]
    return loop


def get_longest_path(graph: ig.Graph) -> np.ndarray:
    """
    Find the longest path in a graph.

    Identifies extremity vertices (local distance maxima) and finds
    the longest shortest path between any pair of extremities.

    Parameters
    ----------
    graph : igraph.Graph
        Input graph with 'coord' vertex attribute.

    Returns
    -------
    path : (N, 3) ndarray
        3D coordinates of vertices forming the longest path.
    """

    # extract point cloud
    coords = graph.vs.get_attribute_values('coord')
    coords = np.row_stack(coords)
    nV = len(coords)

    # find extremities by finding farthest points from random seed
    distances = np.array(graph.distances(0)[0])
    extremities = [0]
    for node in range(1, nV):
        neighbors = graph.neighbors(node)
        d = distances[node]
        neighb_d = distances[neighbors]
        if np.all(d >= neighb_d):
            extremities.append(node)

    max_len, long_path = 0, []
    for ii, u in enumerate(extremities[:-1]):
        for v in extremities[ii+1:]:
            # find shortest path
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path = graph.get_shortest_paths(u, v)[0]
            if len(path) > max_len:
                max_len = len(path)
                long_path = path

    loop = coords[long_path]
    return loop


def redistribute_evenly(
    loop: np.ndarray,
    N: Optional[int] = None
) -> np.ndarray:
    """
    Redistribute points along a loop to achieve uniform spacing.

    Parameters
    ----------
    loop : (M, 3) ndarray
        Input loop coordinates.
    N : int, optional
        Number of output points. Default is len(loop).

    Returns
    -------
    evenly_spaced : (N, 3) ndarray
        Points evenly distributed along the loop.
    """
    loop = loop.copy()
    N = len(loop) if N is None else N
    # Compute total loop length
    distances = np.linalg.norm(loop - np.roll(loop, -1, axis=0), axis=1)
    target_spacing = np.sum(distances) / N

    # Evenly space the points along the smoothed loop
    evenly_spaced_points = np.empty((N, 3), dtype=loop.dtype)
    evenly_spaced_points[0] = loop[0]
    current_distance = 0.0
    current_idx = 0

    orig_N = len(loop)
    for i in range(1, N):
        while current_distance + distances[current_idx] < target_spacing:
            current_distance += distances[current_idx]
            current_idx = (current_idx + 1) % orig_N

        overhang = current_distance + distances[current_idx] - target_spacing
        fraction = 1 - overhang / distances[current_idx]
        evenly_spaced_points[i] = loop[current_idx] * (1 - fraction) + loop[(current_idx + 1) % orig_N] * fraction
        current_distance = 0
        distances[current_idx] = overhang
        loop[current_idx] = evenly_spaced_points[i]

    return evenly_spaced_points


def smooth_loop(loop: np.ndarray, n_iters: int = 3) -> np.ndarray:
    """
    Smooth a loop using iterative neighbor averaging.

    Parameters
    ----------
    loop : (N, 3) ndarray
        Input loop coordinates.
    n_iters : int, optional
        Number of smoothing iterations. Default is 3.

    Returns
    -------
    smoothed : (N, 3) ndarray
        Smoothed loop coordinates.
    """
    for _ in range(n_iters):
        left = np.roll(loop, 1, axis=0)
        right = np.roll(loop, -1, axis=0)

        loop = (left + right) / 2
    return loop
