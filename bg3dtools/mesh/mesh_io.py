"""
Mesh file I/O utilities.

This module provides functions for reading and writing triangle meshes
in various formats (PLY, OBJ, etc.) with support for vertex colors,
normals, and face colors.
"""

from typing import Union, Tuple, Optional
from os.path import isfile
import numpy as np

__all__ = [
    "read_triangle_mesh",
    "read_obj",
    "write_colored_plyfile",
    "read_colored_plyfile",
]


def read_triangle_mesh(
    file: str,
    process: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a triangle mesh from file.

    Parameters
    ----------
    file : str
        Path to mesh file (supports formats: PLY, OBJ, STL, OFF, etc.).
    process : bool, optional
        If True, apply trimesh processing (merge vertices, etc.).
        Default is False.

    Returns
    -------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    Examples
    --------
    >>> verts, faces = read_triangle_mesh('model.ply')
    >>> verts.shape
    (1000, 3)
    """
    if not isfile(file):
        raise FileNotFoundError(f"File not found: {file}")

    import trimesh
    mesh = trimesh.load_mesh(file, process=process)
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        mesh = geometries[0]
    v, f = mesh.vertices, mesh.faces
    v = np.ascontiguousarray(v)
    f = np.ascontiguousarray(f)
    return v, f


def read_obj(
    obj_file: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read an OBJ file with texture coordinates.

    Wrapper for libigl that handles different API versions.

    Parameters
    ----------
    obj_file : str
        Path to OBJ file.

    Returns
    -------
    verts : (nV, 3) ndarray
        Vertex coordinates.
    tc : (nT, 2) ndarray
        Texture coordinates.
    n : (nN, 3) ndarray
        Vertex normals.
    faces : (nF, 3) ndarray
        Face vertex indices.
    ftc : (nF, 3) ndarray
        Face texture coordinate indices.
    fn : (nF, 3) ndarray
        Face normal indices.
    """
    import igl
    try:
        verts, tc, n, faces, ftc, fn = igl.read_obj(obj_file)
    except AttributeError:
        verts, tc, n, faces, ftc, fn = igl.readOBJ(obj_file)

    return verts, tc, n, faces, ftc, fn


# ---------------------------------------------------------------------
#  Write
# ---------------------------------------------------------------------
def write_colored_plyfile(
    outfile: Union[str, bytes],
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    v_rgb: Optional[np.ndarray] = None,
    v_normals: Optional[np.ndarray] = None,
    f_rgb: Optional[np.ndarray] = None,
    text: bool = False
) -> None:
    """
    Save a mesh to PLY with optional colors and normals.

    Parameters
    ----------
    outfile : str or bytes
        Output file path.
    verts : (nV, 3) ndarray
        Vertex coordinates.
    faces : (nF, 3) ndarray
        Triangle indices.
    v_rgb : (nV, 3) ndarray, optional
        Per-vertex RGB colors (uint8, 0-255).
    v_normals : (nV, 3) ndarray, optional
        Per-vertex normals (float32).
    f_rgb : (nF, 3) ndarray, optional
        Per-face RGB colors (uint8, 0-255). Blender-compatible.
    text : bool, optional
        If True, write ASCII PLY. Default is False (binary).

    Examples
    --------
    >>> write_colored_plyfile('output.ply', verts, faces, v_rgb=colors)
    """
    from plyfile import PlyData, PlyElement

    # ---- vertices ----------------------------------------------------
    dtype_fields = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if v_normals is not None:
        dtype_fields += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    if v_rgb is not None:
        dtype_fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    vertex_array = np.empty(verts.shape[0], dtype=dtype_fields)
    vertex_array['x'] = verts[:, 0]
    vertex_array['y'] = verts[:, 1]
    vertex_array['z'] = verts[:, 2]

    if v_normals is not None:
        vn = v_normals.astype(np.float32)
        vertex_array['nx'] = vn[:, 0]
        vertex_array['ny'] = vn[:, 1]
        vertex_array['nz'] = vn[:, 2]

    if v_rgb is not None:
        vc = v_rgb.astype(np.uint8)
        vertex_array['red'] = vc[:, 0]
        vertex_array['green'] = vc[:, 1]
        vertex_array['blue'] = vc[:, 2]

    el_v = PlyElement.describe(vertex_array, 'vertex')

    # ---- faces -------------------------------------------------------
    if f_rgb is None:
        face_array = np.empty(faces.shape[0],
                              dtype=[('vertex_indices', 'i4', (3,))])
        face_array['vertex_indices'] = faces
    else:
        r, g, b = f_rgb.T.astype(np.uint8)
        face_array = np.array(list(zip(faces, r, g, b)),
                              dtype=[('vertex_indices', 'i4', (3,)),
                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el_f = PlyElement.describe(face_array, 'face')

    PlyData([el_v, el_f],
            text=text,
            comments=["written by mesh_io.write_colored_plyfile"]).write(outfile)


# ---------------------------------------------------------------------
#  Read
# ---------------------------------------------------------------------
def read_colored_plyfile(
    infile: Union[str, bytes],
    *,
    vert_colors: bool = False,
    vert_normals: bool = False,
    face_colors: bool = False
) -> Tuple:
    """
    Load a PLY file with optional colors and normals.

    Parameters
    ----------
    infile : str or bytes
        Input file path.
    vert_colors : bool, optional
        If True, return vertex RGB colors if present. Default is False.
    vert_normals : bool, optional
        If True, return vertex normals if present. Default is False.
    face_colors : bool, optional
        If True, return face RGB colors if present. Default is False.

    Returns
    -------
    verts : (nV, 3) ndarray
        Vertex coordinates (float32).
    faces : (nF, 3) ndarray
        Triangle indices (int32).
    v_rgb : (nV, 3) ndarray, optional
        Vertex colors (uint8). Only if vert_colors=True and data exists.
    v_normals : (nV, 3) ndarray, optional
        Vertex normals (float32). Only if vert_normals=True and data exists.
    f_rgb : (nF, 3) ndarray, optional
        Face colors (uint8). Only if face_colors=True and data exists.

    Examples
    --------
    >>> verts, faces = read_colored_plyfile('mesh.ply')
    >>> verts, faces, colors = read_colored_plyfile('mesh.ply', vert_colors=True)
    """
    from plyfile import PlyData

    plydata = PlyData.read(infile)

    vdata = plydata['vertex'].data
    verts = np.vstack([vdata['x'], vdata['y'], vdata['z']]).T.astype(np.float32)

    faces = np.vstack(plydata['face'].data['vertex_indices']).astype(np.int32)

    out = [verts, faces]

    if vert_colors and {'red', 'green', 'blue'}.issubset(vdata.dtype.names):
        v_rgb = np.vstack([vdata['red'], vdata['green'], vdata['blue']]).T.astype(np.uint8)
        out.append(v_rgb)

    if vert_normals and {'nx', 'ny', 'nz'}.issubset(vdata.dtype.names):
        v_n = np.vstack([vdata['nx'], vdata['ny'], vdata['nz']]).T.astype(np.float32)
        out.append(v_n)

    if face_colors and {'red', 'green', 'blue'}.issubset(plydata['face'].data.dtype.names):
        fdata = plydata['face'].data
        f_rgb = np.vstack([fdata['red'], fdata['green'], fdata['blue']]).T.astype(np.uint8)
        out.append(f_rgb)

    return tuple(out)
