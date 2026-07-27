"""
Mesh cleaning and preprocessing pipeline.

Provides the MeshCleaner base class for automated scan cleanup
(hole filling, manifold extraction, floor removal, alignment) and
batch processing utilities.
"""

import logging
from os.path import splitext, isfile
import os
from typing import List, Optional, Tuple, Type

import numpy as np
from scipy.stats import mode
from tqdm import tqdm

from bg3dtools.pointclouds.fitting import fit_plane_to_noisy_points, project_to_plane, align_axes, fit_plane_to_points
from bg3dtools.igl_compat import (
    average_onto_faces,
    cylinder,
    doublearea,
    point_mesh_squared_distance,
    write_triangle_mesh,
)
from bg3dtools.mesh import read_triangle_mesh
from bg3dtools.mesh.utils import submesh, per_vertex_normals, extract_manifold_patches
from bg3dtools.transforms_unified import inverse, transform_points_forward, make_aff
from bg3dtools.utils import HiddenPrints

__all__ = [
    "MalformedFile",
    "NumpyMesh",
    "MeshCleaner",
    "write_clean_ply",
    "clean_scans",
]

# from bg3dtools.render.o3d import scatt, trisurfsm, draw_geometries
class MalformedFile(OSError):
    pass


class NumpyMesh:
    def __init__(self, verts: np.ndarray, faces: np.ndarray) -> None:
        self.verts = verts
        self.faces = faces


class MeshCleaner:
    """
    default MeshCleaner uses IGL backend
    """
    def __init__(self, filename: str, scanner: str = '3dmd', out_suff: str = '_clean.ply') -> None:

        if scanner == '3dmd':
            self.bounding_box = ((-1000, 1000), (-590, 1600), (-1000, 1000))
            self.up_axis = 1
            self.native_to_meters = 0.001
            # platform detection (native mm)
            self.platform_height = 125
            self.plane_fit_thresh = 5
            self.plane_cut_thresh = 20
            # handrail (native mm)
            self.rail_axis = 2
            self.rail_height = 1050
            self.rail_rad = 15
            self.rail_roc = 80
            self.rail_search_margin = 100
            self.icp_inlier_thresh = 5
            # surface area polyfit (native mm² scale)
            self.polyfit = [-170, 53500, 650000]
        elif scanner == '3dfascination':
            self.bounding_box = ((-1, 1), (0.02, 2.2), (-1, 1))
            self.up_axis = 1
            self.native_to_meters = 1.0
            self.platform_height = 0.125
            self.plane_fit_thresh = 0.005
            self.plane_cut_thresh = 0.02
            self.rail_axis = None
            self.polyfit = [-0.00017, 0.0535, 0.65]
        else:
            raise ValueError('%s is not a valid scanner' % scanner)

        self.filename = filename
        self.scanner = scanner
        self.unprocessed = True
        self.log = logging.getLogger('Initializer')
        assert out_suff.endswith('.ply')
        self.out_suff = out_suff

        # poses that include a handrail
        self.leg_lift_poses = ['scan%03d' % i for i in range(21, 35)]
        self.mesh = None

        assert os.path.getsize(filename) > 1024, 'files must be > 1kb'
        self.load()
        v, f = self.numpy_geometry()  # check that mesh is not empty
        assert len(v) > 0, 'empty mesh'
        assert len(f) > len(v), 'too few faces'

    def load(self) -> None:
        v, f = read_triangle_mesh(self.filename)
        self.mesh = NumpyMesh(v, f)

    def write_mesh(self, outfile: Optional[str] = None) -> None:
        if outfile is None:
            basename = splitext(self.filename)[0]
            outfile = basename + self.out_suff

        write_triangle_mesh(outfile, self.mesh.verts, self.mesh.faces)

    @property
    def name(self) -> str:
        return self.filename.split('/')[-3]

    def numpy_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mesh.verts, self.mesh.faces

    def bound_with_box(self) -> None:
        """
        remove faces outside bounding box
        """
        verts, faces = self.numpy_geometry()
        x_lim, y_lim, z_lim = self.bounding_box
        # check if vertices are within bounding box
        v_mask = np.logical_and.reduce((
            verts[:, 0] > x_lim[0], verts[:, 0] < x_lim[1],
            verts[:, 1] > y_lim[0], verts[:, 1] < y_lim[1],
            verts[:, 2] > z_lim[0], verts[:, 2] < z_lim[1]
        ))
        # face mask is true if all vertices are within bounding box
        f_mask = np.logical_and.reduce((
            v_mask[faces[:, 0]], v_mask[faces[:, 1]], v_mask[faces[:, 2]]
        ))
        self.log.debug('  bounding box: kept %d / %d faces' % (np.sum(f_mask), f_mask.size))
        self.submesh(f_mask)

    def submesh(self, f_mask: np.ndarray) -> None:
        new_v, new_f, _, _ = submesh(self.mesh.verts, self.mesh.faces, f_mask)
        self.mesh = NumpyMesh(new_v, new_f)

    def preprocess(self) -> str:
        message = ""
        if self.unprocessed:
            self.bound_with_box()

            platform_plane = None
            if self.scanner == '3dmd':
                platform_plane = self.find_platform()
                message = 'platform detected' if platform_plane is not None else 'no platform'

            if platform_plane is not None:
                self.remove_platform(platform_plane)

                pose_id = self.filename.split('/')[-2]
                if pose_id in self.leg_lift_poses:
                    self.remove_handrail()
                    message += "; handrail removed"

            if len(self.numpy_geometry()[0]) > 0:
                self.remove_loose()

            self.unprocessed = False
        return message

    @staticmethod
    def plausible_size(weight: float, height: float) -> bool:
        plausible_weight = 25 < weight < 150
        plausible_height = 1 < height < 2
        return plausible_height and plausible_weight

    def mostly_intact(self, weight: float, height: Optional[float] = None, thresh: float = 0.75) -> bool:
        """
        compare mesh surface area to expected surface area based on weight
        """
        assert not self.unprocessed, 'mesh must be preprocessed before checking integrity'
        assert 0 < thresh < 1.0, 'threshold must be between 0 and 1'

        # polynomial fit for expected relationship between weight and (doubled) surface area
        p = self.polyfit
        expected_surface_area = p[0] * (weight**2) + p[1] * weight + p[2]
        v, f = self.numpy_geometry()
        sa = np.sum(doublearea(v, f)) if len(f) > 0 else 0

        intact = sa > thresh * expected_surface_area
        self.log.debug('  %s found to be %s' % (self.filename, 'intact' if intact else 'DEFECTIVE'))
        if not intact:
            self.log.debug('    weight=%.2f' % weight)
            self.log.debug('    surface area=%.2f' % sa)
            self.log.debug('    expected surface area=%.2f' % expected_surface_area)

        if intact and height is not None:
            # don't actually enforce height validation, but issue warning
            scan_height = (np.max(v[:, self.up_axis]) - np.min(v[:, self.up_axis])) * self.native_to_meters
            if abs(scan_height - height) / height > 0.05:
                self.log.warning('   %s height does not match clinical value: %.2f (scan) vs %.2f (clinical)' %
                                 (self.filename, scan_height, height))

        return intact

    def find_platform(self, ratio: int = 50) -> Optional[np.ndarray]:
        """
        fit plane at the bottom range of scan area, pointing up
        return None if insufficient best plane has insufficient inliers
        """
        verts = self.numpy_geometry()[0]
        pt_height = verts[:, self.up_axis]
        min_height = np.min(pt_height)
        up_vec = np.zeros(3)
        up_vec[self.up_axis] = 1
        low_mask = pt_height < min_height + (self.platform_height * 1.5)
        low_pts = verts[low_mask]
        plane, inlier_mask = fit_plane_to_noisy_points(low_pts, threshold=self.plane_fit_thresh, target_vec=up_vec, ang_thresh=0.1)

        num_inliers = np.sum(inlier_mask)

        if num_inliers > (verts.shape[0] / ratio):
            self.log.debug('  platform plane fitted with %d inliers' % num_inliers)
            plane = fit_plane_to_points(low_pts[inlier_mask])
            # plane = np.append(normal, -np.dot(normal, pt))
            plane *= 1 if plane[self.up_axis] > 0 else -1
            self.log.debug('plane = %.2f %.2f %.2f %.2f' % (plane[0], plane[1], plane[2], plane[3]))
            return plane
        else:
            self.log.debug('  no platform detected')
            return None

    def remove_platform(self, platform_plane: np.ndarray) -> None:
        # make sure plane is numpy
        plane = np.array(platform_plane, dtype=platform_plane.dtype)

        # remove faces below cut plane
        v, f = self.numpy_geometry()
        bc = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3  # face centroids
        d = project_to_plane(plane, bc)[1]
        # keep only faces whose centroids are above the fitted plane
        face_mask = d > self.plane_cut_thresh

        self.submesh(face_mask)
        self.floor_height = -plane[3] / plane[self.up_axis]
        num_cut = int(np.sum(face_mask))
        self.log.info('  removed platform, kept %d / %d faces' % (num_cut, face_mask.size))

    def remove_handrail(self) -> None:
        if self.rail_axis is None:
            self.log.debug('   rail axis not set; skipping handrail removal')
            return

        rad = self.rail_rad
        abs_rail_height = self.floor_height + self.rail_height
        up_axis, rail_axis = self.up_axis, self.rail_axis

        # third axis (not vertical or aligned with handrail)
        off_axis = 3 - up_axis - rail_axis
        v, f = self.numpy_geometry()
        vidx = np.arange(len(v))

        # Step 1: find vertical pole(s)
        # extract points below height
        height = v[:, up_axis]
        h_mask = height < abs_rail_height + 2*rad
        pts, vidx = v[h_mask], vidx[h_mask]

        # restrict search to points at the edge of the scan (where handrail should be)
        edge_mask = self.get_edge_mask(pts, up_axis, off_axis, abs_rail_height)
        # note: these points could be the handrail, or a leg!
        pts, vidx = pts[edge_mask], vidx[edge_mask]

        # search for vertical section of pole
        low_mask = pts[:, up_axis] < abs_rail_height - self.rail_search_margin
        if len(pts < 500) or np.sum(low_mask) == 0:
            self.log.info('   no handrail detected')
            return
        ver_pt = np.median(pts[low_mask], axis=0)

        # search for horizontal section of pole
        low_cut = pts[:, up_axis] > abs_rail_height - 2*rad
        high_cut = pts[:, up_axis] < abs_rail_height + 2*rad
        horizontal_mask = np.logical_and(low_cut, high_cut)
        if np.sum(horizontal_mask) == 0:
            self.log.info('   no handrail detected')
            return
        # find point farthest from vertical pole
        hor_pts = pts[horizontal_mask]
        i = np.argmax(np.abs(hor_pts[:, rail_axis] - ver_pt[rail_axis]))
        hor_pt = hor_pts[i]

        rail_mask, ver_mag = self.register_handrail(pts, ver_pt, hor_pt)
        nI = np.sum(rail_mask)

        if nI > 400 and ver_mag > 0.95:
            self.log.info('   remove handrail with %d vertices' % nI)
            full_mask = np.ones(len(v), dtype=bool)
            vidx = vidx[rail_mask]
            full_mask[vidx] = False
            f_mask = np.round(average_onto_faces(f, full_mask.astype(np.float32)))
            self.submesh(f_mask)
        else:
            self.log.info('   no handrail detected')

    def remove_loose(self) -> None:
        f = self.numpy_geometry()[1]
        p = extract_manifold_patches(f)[1]
        # f_mask = p == mode(p, keepdims=False)[0]  # numpy version?
        f_mask = p == mode(p)[0]
        self.submesh(f_mask)

    def get_edge_mask(self, pts: np.ndarray, up_axis: int, off_axis: int, rail_height: float) -> np.ndarray:
        # extract points on edge of scan (where handrail would be)
        margin = self.rail_search_margin
        edginess = pts[:, off_axis]
        low_mask = pts[:, up_axis] < rail_height - margin / 2
        low_pts = pts[low_mask]
        emask_1 = edginess < np.min(low_pts[:, off_axis]) + margin
        emask_2 = edginess > np.max(low_pts[:, off_axis]) - margin
        edge_mask = emask_1 if np.sum(emask_1) > np.sum(emask_2) else emask_2
        return edge_mask

    def register_handrail(self, pts: np.ndarray, ver_cog: np.ndarray, hor_cog: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        align template model with scan points
        do this by moving the scan to match the template, because we trust the template normals more than scan normals
        then invert the transform to find the template in scan coordinates
        """
        from simpleicp import PointCloud, SimpleICP

        # compute initial transformation: move scan to origin (where template is built)
        pt0 = ver_cog.copy()
        pt0[self.up_axis] = hor_cog[self.up_axis]
        pt1, pt2 = ver_cog, hor_cog
        init_tform = inverse(align_axes(pt0, pt1, pt2))
        pts_init = transform_points_forward(init_tform, pts)

        # convert pts to PointCloud object
        pc_mov = PointCloud(pts_init, columns=["x", "y", "z"])

        # construct template handrail (points and normals)
        template_v, template_normals, template_f = self.construct_handrail()
        pc_fix = PointCloud(np.column_stack([template_v, template_normals]), columns=["x", "y", "z", "nx", "ny", "nz"])

        # Create simpleICP object, add point clouds, and run algorithm!
        icp = SimpleICP()
        icp.add_point_clouds(pc_fix, pc_mov)
        with HiddenPrints():
            H, X_mov_transformed, _, _ = icp.run(max_overlap_distance=self.rail_height)

        # parse outputs
        d2 = point_mesh_squared_distance(X_mov_transformed, template_v, template_f)[0]
        inlier_mask = d2 < self.icp_inlier_thresh**2
        final_tform = H @ init_tform
        vertical_mag = abs(final_tform[0, self.up_axis])

        return inlier_mask, vertical_mag

    def construct_handrail(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rail_rad = self.rail_rad
        rail_height = self.rail_height
        rail_roc = self.rail_roc

        # straight pole
        v, f = cylinder(64, 1024)
        nV = len(v)
        v[:, [0, 1]] *= rail_rad
        v[:, 2] -= 0.5
        v[:, 2] *= 2 * (rail_height - rail_roc)
        # align to x axis
        v = v[:, [2, 0, 1]]
        # bend down to y axis
        h = v[:, 0]
        bweight = ((-h / rail_roc).clip(-1, 1) + 1) / 2
        bweight = bweight.reshape((-1, 1, 1))

        # move verts in anticipation of bending around axis
        v[:, 1] -= rail_roc

        # linear blend skin
        rotm = make_aff([0, 0, np.pi/2], [0, 0, 0])   # bend, then restore to original location

        G = np.tile(rotm, (nV, 1, 1)) * bweight
        E = np.tile(np.eye(4), (nV, 1, 1)) * (1 - bweight)
        G_w = G + E
        v_h = np.hstack((v, np.ones((nV, 1))))
        posed_homog = (G_w[:, 0, :] * v_h[:, 0, None] + G_w[:, 1, :] * v_h[:, 1, None] +
                       G_w[:, 2, :] * v_h[:, 2, None] + G_w[:, 3, :] * v_h[:, 3, None])
        posed_homog /= posed_homog[:, 3, None]
        v = posed_homog[:, 0:3]

        v += [rail_roc, rail_roc, 0]
        normals = per_vertex_normals(v, f)
        return v, normals, f


def write_clean_ply(outfile: str, verts: np.ndarray, faces: np.ndarray, scale: float = 0.001) -> None:
    """Write a clean mesh PLY with scale metadata in the header.

    Args:
        outfile: Output file path.
        verts: (N, 3) vertex positions (already in target units).
        faces: (F, 3) triangle indices.
        scale: Native-to-meters conversion factor applied to produce these
               vertices.  Stored as ``comment scale:<value>`` in the PLY header.
    """
    from plyfile import PlyData, PlyElement

    vertex_data = np.array(
        list(zip(verts[:, 0], verts[:, 1], verts[:, 2])),
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')],
    )
    el_v = PlyElement.describe(vertex_data, 'vertex')

    ply_faces = np.empty(faces.shape[0], dtype=[('vertex_indices', 'i4', (3,))])
    ply_faces['vertex_indices'] = faces
    el_f = PlyElement.describe(ply_faces, 'face')

    plydata = PlyData([el_v, el_f], text=False, comments=[f'scale:{scale}'])
    plydata.write(outfile)


def clean_scans(
    scan_files: List[str],
    Cleaner: Type[MeshCleaner],
    height: Optional[float] = None,
    weight: Optional[float] = None,
    skip_existing: bool = False,
    stop_on_fail: bool = False,
    permissivity: float = 1.0,
    scanner: str = '3dmd',
    in_suff: str = '.obj',
    out_suff: str = '_clean.ply',
) -> List[bool]:
    log = logging.getLogger('clean_scans')

    if (height and weight) and np.all(np.isfinite([height, weight])):
        assert MeshCleaner.plausible_size(weight, height), 'implausible height (%.2f) or weight (%.2f)' % (height, weight)

    # pre-processing
    success = [False] * len(scan_files)
    threshold = 0.75 ** permissivity

    progress = tqdm(scan_files, desc='Preprocessing scans')
    for ii, filename in enumerate(progress):
        outfile = filename.replace(in_suff, out_suff)
        defectfile = filename.replace(in_suff, '.bad')
        assert outfile != filename, 'output file must differ from input file'

        if skip_existing and (isfile(outfile) or isfile(defectfile)):
            success[ii] = isfile(outfile)
            log.debug('  %s already processed; skipping' % filename)
            continue

        log.debug('preprocessing %s' % filename)
        if isfile(outfile): os.remove(outfile)
        if isfile(defectfile): os.remove(defectfile)

        try:
            cleaner = Cleaner(filename, scanner=scanner)
            message = cleaner.preprocess()
            progress.set_postfix_str(message)

            if (weight is not None) and not cleaner.mostly_intact(weight, height=height, thresh=threshold):
                raise MalformedFile('  %s failed size check' % filename)

            success[ii] = True
            cleaner.write_mesh(outfile)

        except MalformedFile as e:
            with open(defectfile, mode='a'): pass
            log.error(e)

        except Exception as e:
            log.error('  %s failed preprocessing: %s' % (filename, e))
            if stop_on_fail:
                raise e
            continue

    return success


