"""
Graph operations for skeleton analysis.

This module provides functions for converting 3D skeletons to graphs
and extracting paths and loops from graph structures.
"""

import logging
import warnings
from collections import deque
from typing import Optional

import numpy as np
import scipy.spatial as spatial
import igraph as ig

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

    voxel_pts = np.argwhere(skeleton).astype(np.float64)
    pts_scaled = convert_to_points(skeleton, affine)

    tree = spatial.cKDTree(voxel_pts, leafsize=8)
    edges = list(tree.query_pairs(r=1.75, p=2))  # 26-connected in voxel space

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


def get_all_loops(
    graph: ig.Graph,
    min_cycle_len: int = 10,
) -> tuple[list[list[int]], np.ndarray]:
    """Find all fundamental cycles in a graph.

    O(V+E) BFS spanning tree + LCA cycle reconstruction.

    Parameters
    ----------
    graph : igraph.Graph
        Input graph with 'coord' vertex attribute.
    min_cycle_len : int, optional
        Minimum number of vertices in a cycle to keep. Cycles shorter than
        this are discarded before path reconstruction (cheap length estimate
        via BFS depths). Set to 0 to return all cycles. Default is 10.

    Returns
    -------
    cycles : list of list[int]
        Vertex indices for each cycle, sorted by length (largest first).
        Empty list if no cycles exist.
    coords : (V, 3) ndarray
        All vertex coordinates.
    """
    coords = np.row_stack(graph.vs.get_attribute_values('coord'))
    n = len(coords)

    if n == 0:
        return [], coords

    # BFS spanning tree from vertex 0
    parent = np.full(n, -1, dtype=np.int32)
    depth = np.zeros(n, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    tree_edges = set()

    root = 0
    visited[root] = True
    queue = deque([root])

    while queue:
        u = queue.popleft()
        for v in graph.neighbors(u):
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                tree_edges.add((min(u, v), max(u, v)))
                queue.append(v)

    # identify non-tree edges (each defines one fundamental cycle)
    all_edges = set()
    for e in graph.get_edgelist():
        all_edges.add((min(e[0], e[1]), max(e[0], e[1])))

    non_tree_edges = all_edges - tree_edges

    # for each non-tree edge, find cycle via LCA
    all_cycles = []

    for u, v in non_tree_edges:
        # skip if either endpoint wasn't reached by BFS (disconnected)
        if not visited[u] or not visited[v]:
            continue

        # find LCA by walking both up to same depth, then walking together
        a, b = u, v

        # bring deeper node up to same depth
        while depth[a] > depth[b]:
            a = parent[a]
        while depth[b] > depth[a]:
            b = parent[b]

        # walk both up until they meet
        while a != b:
            a = parent[a]
            b = parent[b]
        lca = a

        # cheap cycle length estimate before full reconstruction
        cycle_len = int(depth[u]) + int(depth[v]) - 2 * int(depth[lca]) + 1
        if cycle_len < min_cycle_len:
            continue

        # reconstruct cycle: u -> lca, then lca -> v (reversed)
        path_u = []
        a = u
        while a != lca:
            path_u.append(a)
            a = parent[a]
        path_u.append(lca)

        path_v = []
        b = v
        while b != lca:
            path_v.append(b)
            b = parent[b]
        # path_v goes v -> lca, we want lca -> v, so reverse
        # full cycle: u -> lca -> v  (path_u + reversed path_v without lca)
        path_v.reverse()
        cycle = path_u + path_v

        all_cycles.append(cycle)

    # sort by length, largest first
    all_cycles.sort(key=len, reverse=True)

    return all_cycles, coords


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


def get_longest_path_fast(graph: ig.Graph) -> np.ndarray:
    """Find the longest shortest path using double-BFS diameter heuristic.

    Two BFS passes find approximate diameter endpoints in O(V+E),
    replacing the O(K^2 * (V+E)) extremity-pair search.  Exact on
    trees; near-optimal on sparse skeleton graphs.

    Parameters
    ----------
    graph : igraph.Graph
        Input graph with 'coord' vertex attribute.

    Returns
    -------
    path : (N, 3) ndarray
        3D coordinates of vertices forming the longest path.
    """
    coords = np.row_stack(graph.vs.get_attribute_values('coord'))
    n = len(coords)
    if n == 0:
        return np.empty((0, 3))

    # BFS from vertex 0 → farthest vertex u
    d0 = np.array(graph.distances(0)[0])
    u = int(np.argmax(d0))

    # BFS from u → farthest vertex v
    du = np.array(graph.distances(u)[0])
    v = int(np.argmax(du))

    # reconstruct path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = graph.get_shortest_paths(u, v)[0]

    return coords[path]


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
