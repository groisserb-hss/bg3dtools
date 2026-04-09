"""
Mesh class for spectral shape analysis.

Provides a mesh data structure with lazy-computed Laplacian
eigendecomposition, geodesic distances, mass matrix, and spectral
filtering operations for functional map pipelines.
"""

# Derived from pyFM by Robin Magnet (MIT License) — see /THIRD_PARTY_NOTICES.txt
import json
import logging
import os
from time import time

import igl
import numpy as np
from scipy import sparse

from . import geometric_utilities as util
from bg3dtools.mesh.mesh_io import read_triangle_mesh
from bg3dtools.mesh.laplace import laplace_eigen_decomposition, gaussian_curvature
from bg3dtools.mesh.laplace import cotangent_weights, fem_mass_matrix

log = logging.getLogger(__name__)

""" ================================================================================= """
"""                         Mesh Class Definition                                     """
""" ================================================================================= """


class Mesh:
    def __init__(
        self,
        v: np.ndarray = np.array([]),  # Vertices
        f: np.ndarray = np.array([]),  # Faces
        g: np.ndarray = np.array([]),  # Geodesic Matrix
        s: np.ndarray = np.array([]),  # Signature Functions
        num_eigenvectors: int = 100,  # Number of Eigenvectors and values
        type: str = "",
    ) -> None:
        self.__v = v  # Vertices
        self.__f = f  # Faces
        self.__g = g  # Geodesic Matrix
        self.__s = s  # Signature Functions
        self.__num_eigenvectors = (
            num_eigenvectors  # Number of Eigenvectors and values
        )
        self.__l: np.ndarray | sparse.spmatrix = np.array([])  # Laplacian
        self.__mass: np.ndarray | sparse.spmatrix = np.array([])
        self.__normals: np.ndarray = np.array([])
        self.__eigen: list[np.ndarray] = [np.array([]) for _ in range(2)]
        # self.__scalars = {}
        self.__type = type
        self.__name = ""

    @staticmethod
    def from_file(fn: str, normalize: bool = False) -> "Mesh":
        if fn[-4:] == ".npz":
            data = dict(np.load(fn))
            v = data["v"]
            f = data["f"]
            g = data["g"] if "g" in data else np.array([])
            s = data["s"] if "s" in data else np.array([])
            num_eigenvectors = data["num_eigenvectors"]
            type = data["type"]
            return Mesh(v, f, g, s, num_eigenvectors, type)
        else:
            v, f = read_triangle_mesh(fn)
            if normalize:
                v, f = util.normalize_mesh(v, f)
            return Mesh(v, f)

    def load(self, fn: str) -> "Mesh":
        v, f = read_triangle_mesh(fn)
        self.__v = v
        self.__f = f
        return self

    def save_np(self, fn: str) -> None:
        data = {"v": self.v,
                "f": self.f,
                "num_eigenvectors": self.num_eigenvectors,
                "type": self.type,
                "name": self.name}
        if self.g.size > 0:
            data["g"] = self.g
        if self.s.size > 0:
            data["s"] = self.s

        np.savez(fn, **data)

    def __getitem__(self, idx: np.ndarray) -> "Mesh":
        if idx.size == self.num_vertices():
            v, f = util.reorder_mesh(self.v, self.f, idx)
        else:
            v = self.v[idx]
            f = np.array([])

        g = np.array([]) if (self.__g.size == 0) else self.g[idx][:, idx]
        s = np.array([]) if (self.__s.size == 0) else self.s[idx]
        res = Mesh(
            v,
            f,
            g=g,
            s=s,
            num_eigenvectors=self.num_eigenvectors,
            type=self.type,
        )
        # for fn in self.scalars:
        #     try:
        #         res.scalars[fn] = self.scalars[fn][idx].copy()
        #     except Exception:
        #         pass

        return res

    def copy(self) -> "Mesh":
        res = Mesh(
            v=self.__v.copy(),
            f=self.__f.copy(),
            g=self.__g.copy(),
            s=self.__s.copy(),
            num_eigenvectors=self.__num_eigenvectors,
            type=self.__type,
        )
        res.mass = self.__mass.copy()
        res.normals = self.__normals.copy()
        res.eigen = [e.copy() for e in self.__eigen]
        res.name = self.name
        # for fn in self.scalars:
        #     res.scalars[fn] = self.scalars[fn].copy()
        return res

    """ ================================================ """
    """                     Properties                   """
    """ ================================================ """

    def area(self) -> np.floating:
        return util.area(self.v, self.f)

    def num_vertices(self) -> int:
        """
        Returns the number of vertices of a mesh.

        Returns
        -------
            Number of vertices of mesh
        """
        return self.__v.shape[0]

    """ ================================================ """
    """                 Getters and Setters              """
    """ ================================================ """
    # Vertices
    @property
    def v(self) -> np.ndarray:
        return self.__v

    # Vertices
    @v.setter
    def v(self, v: np.ndarray) -> None:
        self.__v = v

    # Faces
    @property
    def f(self) -> np.ndarray:
        return self.__f

    # Faces
    @f.setter
    def f(self, f: np.ndarray) -> None:
        self.__f = f

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str) -> None:
        self.__name = name

    @property
    def type(self) -> str:
        return self.__type

    # @property
    # def scalars(self):
    #     return self.__scalars
    #
    # @scalars.setter
    # def scalars(self, scalars):
    #     self.__scalars = scalars

    @property
    def num_eigenvectors(self) -> int:
        return self.__num_eigenvectors

    @num_eigenvectors.setter
    def num_eigenvectors(self, k: int) -> None:
        self.__eigen = (
            [e[..., :k] for e in self.__eigen]
            if (k <= self.__num_eigenvectors)
            else [np.array([]) for _ in range(2)]
        )
        self.__num_eigenvectors = k

    # Geodesic Matrix
    @property
    def g(self) -> np.ndarray:
        if self.__g.size == 0:
            t0 = time()
            self.__g = util.geodesic_matrix(self.v, self.f)
            log.info('geodesic matrix: %d verts, %.2fs', len(self.__v), time() - t0)
        return self.__g

    # Geodesic Matrix
    @g.setter
    def g(self, geodesicmatrix: np.ndarray) -> None:
        self.__g = geodesicmatrix

    @property
    def s(self) -> np.ndarray:
        return self.__s

    @s.setter
    def s(self, s: np.ndarray) -> None:
        assert s.shape[0] == self.num_vertices()
        self.__s = s

    @property
    def gaussian(self) -> np.ndarray:
        if self.__gaussian.size == 0:
            self.__gaussian = gaussian_curvature(self.v, self.f)
        return self.__gaussian

    @gaussian.setter
    def gaussian(self, gaussian: np.ndarray) -> None:
        self.__gaussian = gaussian

    @property
    def normals(self) -> np.ndarray:
        if self.__normals.size == 0:
            self.__normals = igl.per_vertex_normals(
                self.__v, self.__f, weighting=0
            )
        return self.__normals

    @normals.setter
    def normals(self, normals: np.ndarray) -> None:
        self.__normals = normals

    @property
    def eigen(self) -> list[np.ndarray]:
        if any([e.size == 0 for e in self.__eigen]):
            t0 = time()
            self.__eigen = laplace_eigen_decomposition(
                self.l, self.mass, self.num_eigenvectors
            )
            log.info('eigen decomposition: %d verts, k=%d, %.2fs',
                     len(self.__v), self.num_eigenvectors, time() - t0)
        return self.__eigen

    @eigen.setter
    def eigen(self, eigen: list[np.ndarray]) -> None:
        self.__num_eigenvectors = eigen[0].size
        self.__eigen = eigen
        self._mass_eigen_cache = None

    @property
    def mass(self) -> sparse.spmatrix:
        if self.__mass.size == 0:
            # igl.massmatrix is broken in igl 2.5.1 (returns all zeros)
            self.__mass = fem_mass_matrix(self.__v, self.__f)
        return self.__mass

    @mass.setter
    def mass(self, mass: np.ndarray | sparse.spmatrix) -> None:
        self.__mass = mass
        self._mass_eigen_cache = None

    # Laplacian
    @property
    def l(self) -> sparse.spmatrix:
        if self.__l.size == 0:
            # igl.cotmatrix is broken in igl 2.5.1 (returns all zeros)
            self.__l = -cotangent_weights(self.v, self.f)
        return self.__l

    # Laplacian
    @l.setter
    def l(self, laplacian: np.ndarray | sparse.spmatrix) -> None:
        self.__l = laplacian

    """ ================================================ """
    """                Scalar Conversion                 """
    """ ================================================ """

    def pointwise_2_vector(self, scalar: np.ndarray, k: int = -1) -> np.ndarray:
        # Cache mass @ eigenvectors to avoid recomputing each call
        if not hasattr(self, '_mass_eigen_cache') or self._mass_eigen_cache is None:
            self._mass_eigen_cache = self.mass @ self.eigen[-1]
        evecs = self._mass_eigen_cache
        if 0 < k:
            evecs = evecs[:, :k]
        return evecs.T @ scalar

    def vector_2_pointwise(self, vector: np.ndarray) -> np.ndarray:
        _, evecs = self.eigen
        k = vector.shape[0]
        return evecs[:, :k] @ vector

    def filter(self, scalar: np.ndarray, k: int = -1) -> np.ndarray:
        s = self.pointwise_2_vector(scalar, k)
        return self.vector_2_pointwise(s)

    def filter_array(self, array: np.ndarray, k: int = -1) -> np.ndarray:
        array = self.filter(array, k=k)
        return self.filter(array.T, k=k).T

    def dirac_deltas(self, i: np.ndarray, k: int = -1) -> np.ndarray:
        x = np.zeros((self.num_vertices(), i.size))
        j = np.arange(i.size)
        x[i, j] = 1
        return self.pointwise_2_vector(x, k)

    """ ================================================ """
    """                Inplace Modifiers                 """
    """ ================================================ """

    def scale(self, factor: float) -> "Mesh":
        self.__v *= factor
        return self

    def rotate(self, R: np.ndarray) -> "Mesh":
        self.__v = self.v @ R
        return self

    def shift(self, *args: float | np.ndarray) -> "Mesh":
        dv: np.ndarray
        if 1 == len(args):
            (dv,) = args
        elif 3 == len(args):
            dx, dy, dz = args
            dv = np.array([dx, dy, dz], dtype=self.__v.dtype)
        else:
            raise Exception("wrong number of arguments. Must be 1 or 3.")
        self.__v += dv
        return self

    def centroid(self) -> np.ndarray:
        return np.mean(self.v, axis=0)

    def centre(self) -> "Mesh":
        self.__v -= self.centroid()
        return self

    def centre_on_boundary(self) -> None:
        b = util.boundary_vertices(self.f)
        temp = igl.boundary_loop(self.f)
        assert all([v in temp for v in b])
        self.__v -= self.v[b].mean()
        return

    def normalise(self) -> "Mesh":
        norm = np.sqrt(self.area())
        self.__v /= norm
        self.__g /= norm
        return self

    """ ================================================ """
    """                     Display                      """
    """ ================================================ """

    def display(self, **kwargs) -> None:
        import vedo as vp

        msh = self.vedo()

        if "scalar" in kwargs:
            s = kwargs["scalar"]
            if isinstance(s, np.ndarray):
                msh.cmap("jet", s)
            elif s in self.scalars:
                msh.cmap("jet", self.scalars[s])
            else:
                raise ValueError("Argument not recognised.")
            msh.add_scalarbar()
            # msh.addScalarBar()

        if "alpha" in kwargs:
            msh.alpha(kwargs["alpha"])

        actors = [msh]

        if "i" in kwargs:
            i = kwargs["i"]
            # planes   = ['r', 'g', 'b', 'k', 'y', 'cyan']
            sph = vp.shapes.Spheres(self.v[i], r=0.01)
            actors.append(sph)

        fig = None
        if "fig" not in kwargs:
            fig = vp.plotter.Plotter()
        else:
            fig = kwargs["fig"]

        fig.add(actors)
        fig.show()

    def vedo(self, c: str | None = None):
        import vedo as vp

        return vp.Mesh([self.v, self.f], c=c)

    def from_vedo(self, vedo_mesh) -> "Mesh":
        self.__v = vedo_mesh.vertices.copy()
        self.__f = np.asarray(vedo_mesh.cells)
        return self

    def write(self, fn: str, method: str = "vedo") -> bool:
        ret = False
        if method == "igl":
            ret = igl.write_triangle_mesh(fn, self.v, self.f)
        else:
            self.vedo().write(fn)
            ret = True
        return ret

    """ ================================================ """
    """                  End of Class                    """
    """ ================================================ """


def read_config(dir: str) -> dict:
    config: dict
    fn = os.path.join(dir, "config.json")
    with open(fn) as file:
        config = json.load(file)
    return config
