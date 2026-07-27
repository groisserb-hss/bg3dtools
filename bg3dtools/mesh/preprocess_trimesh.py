"""
Trimesh-backed mesh cleaning pipeline.

Extends the base MeshCleaner with a trimesh I/O backend for loading,
cleaning, and exporting scanned meshes.
"""

import logging
from os.path import join, splitext, expanduser
from argparse import ArgumentParser
from glob import glob
import sys

import trimesh
import numpy as np
from scipy.stats import mode

from bg3dtools.igl_compat import remove_duplicate_vertices
from bg3dtools.mesh.preprocess import MeshCleaner, clean_scans
from bg3dtools.mesh.utils import extract_manifold_patches

__all__ = [
    "MeshCleanerTrimesh",
]


class MeshCleanerTrimesh(MeshCleaner):

    def load(self):
        self.mesh = trimesh.load(self.filename)

    def numpy_geometry(self):
        verts = self.mesh.vertices
        faces = self.mesh.faces
        return verts, faces

    def submesh(self, f_mask):
        fidx = np.argwhere(f_mask).flatten()
        if len(fidx) == 0:
            self.mesh = trimesh.Trimesh()
        else:
            from trimesh.util import log as trilog
            l = trilog.level
            trilog.setLevel(logging.FATAL)
            self.mesh = self.mesh.submesh([fidx], append=True)
            trilog.setLevel(l)

    def write_mesh(self, outfile=None):
        basename = splitext(self.filename)[0]
        if outfile is None:
            outfile = basename + self.out_suff
            extension = splitext(self.out_suff)[1]
        else:
            extension = splitext(outfile)[1]

        assert extension in ['.stl', '.off', '.ply', '.collada', '.json', '.dict', '.glb', '.dict64', '.msgpack']

        if extension != '.ply':
            self.mesh.export(outfile)
        self.mesh.export(outfile.replace(extension, '.ply'))  # anonymized (no RGB texture)

    def remove_loose(self):
        # trimesh splits vertices at texture boundaries, so stitch them back together
        # before looking for loose patches. Face count/order is preserved by
        # remove_duplicate_vertices, so patch labels map directly to original faces.
        v, f = self.numpy_geometry()
        sv, svi, svj, sf = remove_duplicate_vertices(v, f, 0.000001)
        p = extract_manifold_patches(sf)[1]
        f_mask = p == mode(p, keepdims=False)[0]
        self.submesh(f_mask)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pattern', type=str, required=True)
    parser.add_argument('--height', type=float, required=True)
    parser.add_argument('--weight', type=float, required=True)
    parser.add_argument('--skip', action='store_true')

    # parse input arguments
    config = parser.parse_args()
    # logfile = expanduser(config.log_file)
    logging.basicConfig(level=logging.INFO, force=True,
                        handlers=[logging.StreamHandler(sys.stderr)])

    scans = glob(expanduser(config.pattern))
    scans.sort()
    clean_scans(scans, MeshCleanerTrimesh, config.height, config.weight, config.skip, out_suff='_clean.glb')

