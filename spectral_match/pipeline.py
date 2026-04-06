"""
End-to-end spectral mesh matching pipeline.

Orchestrates mesh preprocessing, spectral decomposition, descriptor
computation, functional map solving, and dense correspondence
extraction between a source and target mesh.
"""

import logging
import numpy as np
import igl
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from bg3dtools.mesh.clean import make_manifold
from bg3dtools.mesh.modify import resize_to_num_verts
from bg3dtools.mesh.utils import mesh_volume
from bg3dtools.mesh.utils import bc2sparse
from bg3dtools.mesh.utils import surface_sample, per_vertex_normals, per_vertex_smoothing, project_to_bccoord
from bg3dtools.mesh.registration import rigid_ICP, nonrigid_ICP

from spectral_match.correspondence.product_manifold_filters import product_manifold_filter as pmf
from spectral_match.tools.mesh_class import Mesh
from spectral_match.correspondence.feature_descriptors import DescriptorClass
from spectral_match.correspondence.deep_functional_maps.operations import ResidualNet
from spectral_match.correspondence.match import compute_correspondence
from spectral_match import default_sig_config, default_match_config

# for debugging
from spectral_match.tools.util import double_plot, vcolor
from bg3dtools.render.trimesh import trisurfsm, scatt, scatts, draw_geometries, draw_axes

def pmf_match(src_hr, dst_orig, config, src=None, dst=None):
    """
    Match two meshes using euclidean initilization and product manifold filter optimization
    :param src:
    :param dst:
    :param config:
    :return:

    Note: all points on dst (bone) are matched to a point on src (template), but not vice versa
    """

    assert len(dst.v) <= len(src.v), 'destination mesh must have fewer vertices than source mesh'
    # initialize with euclidean matching
    aligned_v = rigid_ICP(dst.v, src.f, src.v, scale=True)[1]
    map = surface_sample(src.v, src.f, N=30000)[0]
    p = map @ src.v
    n = map @ per_vertex_normals(src.v, src.f)
    aligned_v = nonrigid_ICP(p, dst.f, aligned_v, pt_normals=n, model_weight=3.)[0]

    D = cdist(aligned_v, src.v)
    j = np.argmin(D, axis=1)  # index into dst for each vertex in src
    i = np.arange(len(dst.v))  # index into src

    assign = lambda x: linear_sum_assignment(x, maximize=True)
    i, j = pmf.product_manifold_filter_correspondence(assign, dst.g, src.g, i, j, config)
    dg = np.abs(src.g[j][:, j] - dst.g[i][:, i]).mean()
    query_on_template = src.v[j]

    for _ in range(10):
        query_on_template = per_vertex_smoothing(query_on_template, dst.f)
        query_on_template = igl.point_mesh_squared_distance(query_on_template, src_hr.v, src_hr.f)[2]

    # upsample the query to the original mesh resolution
    up_map = project_to_bccoord(dst.v, dst.f, dst_orig.v, return_map=True)[2]
    query_on_template = up_map @ query_on_template

    # c = vcolor(dst_orig.v)
    # double_plot(dst_orig, Mesh(query_on_template, dst_orig.f), cmap1=c, cmap2=c)

    template2query, query2template = TemplateMapper.cartesian_mapping(src_hr.v, src_hr.f, query_on_template, dst_orig.f)

    return dg, query2template, template2query


class FunctionalMapper:
    """
    Combine all the necessary functions into one class

    number_vertices: 2000 # default range is 1000 - 5000.
    num_eigs: 75 # default range is 75 - 150
    number_hks: 25  # default range is 100 - 200
    number_wks: 75 # default range is 100 - 200
    number_gaussian: 18 # default range is 10 - 20

    initial_solve_dimension: 8 # solve dimension prior to zoomout refinement
    symmetry_optimisation: true # set to true if the mesh has an intrinsic symmetry
    product_manifold_filter: # the following quantities are relative to the average geodesic distance
        sigma: 0.75
        gamma: 0.5 # sigma is reduced by this factor at each iteration
        iterations: 2
    """
    def __init__(self, target_size=3200,
                 num_eigs=75,
                 num_layers=7,
                 weight_file=None,
                 sig_config=None,
                 match_config=None,
                 extra_sig_fun=None):

        self.logger = logging.getLogger('FunctionalMapper')

        self.target_size = target_size
        self.sig_config = default_sig_config if sig_config is None else sig_config
        self.match_config = default_match_config if match_config is None else match_config
        self.num_eigenvectors = num_eigs
        self.num_layers = num_layers
        self.num_signatures = self.sig_config.num_signatures

        # class that computes signature functions
        self.describer = DescriptorClass(self.sig_config)
        self.extra_sig_fun = extra_sig_fun

        self.resnet = None
        if weight_file is not None:
            self.load_resnet(weight_file)

    def load_resnet(self, weight_file):
        self.resnet = ResidualNet(self.num_layers, self.num_signatures, training=False)
        self.resnet.load_weights(weight_file).expect_partial()

    def preprocess_mesh(self, verts: np.ndarray, faces: np.ndarray, target_size=None) -> Mesh:

        self.logger.debug('Preprocessing query mesh; making manifold')
        verts, faces = make_manifold(verts, faces)

        # decimate to args.target_size
        N = self.target_size if target_size is None else target_size
        self.logger.debug('resize to %d vertices' % N)
        verts, faces = resize_to_num_verts(verts, faces, N)

        mesh = Mesh(verts, faces, num_eigenvectors=self.num_eigenvectors)
        assert np.size(mesh.g) > 0, 'Failed to compute geodesic distances'
        self.logger.debug('Computed geodesic distances')

        # compute eigenvalues and eigenvectors
        assert all([np.size(e) > 0 for e in mesh.eigen]), 'Failed to compute eigenvectors'
        self.logger.debug('Computed eigenvectors')

        # compute signature functions
        if self.describer is not None:
            mesh.s = self.describer(mesh)
            logging.debug('Computed %d signature functions' % mesh.s.shape[1])

        return mesh

    def mesh_correspondence(self, src: Mesh, dst: Mesh, initial_dimensions=-1, use_deep_maps=True):
        initial_dimensions = self.match_config.initial_solve_dimension if initial_dimensions < 0 else initial_dimensions

        # tweak signature functions to optimize functional mapping
        src = src.copy()
        dst = dst.copy()
        if use_deep_maps and self.resnet is not None:
            raw1 = src.s[np.newaxis].astype(np.float32)
            src.s = self.resnet(raw1).numpy()[0]
            raw2 = dst.s[np.newaxis].astype(np.float32)
            dst.s = self.resnet(raw2).numpy()[0]

        if self.extra_sig_fun is not None:
            top_val = max(np.max(src.s), np.max(dst.s))
            src.s = np.column_stack((src.s, self.num_signatures * 100 * top_val * self.extra_sig_fun(src)))
            dst.s = np.column_stack((dst.s, self.num_signatures * 100 * top_val * self.extra_sig_fun(dst)))

        # convert to format expected by subroutines
        pmf_config = {"sigma": self.match_config.pmf_sigma,
                      "gamma": self.match_config.pmf_gamma,
                      "iterations": self.match_config.pmf_iters}
        match_config = {"initial_solve_dimension": initial_dimensions,
                        "symmetry_optimisation": self.match_config.symmetry_optimisation,
                        "product_manifold_filter": pmf_config,
                        "euclidean_init": self.match_config.euclidean_init}

        i, j = compute_correspondence(src, dst, match_config)

        return i, j


class TemplateMapper:
    def __init__(self, fmapper: FunctionalMapper, template_v: np.ndarray, faces: np.ndarray,
                 processed=None, exhaustive_search=True, symmetry_axis=None):

        self.fmapper = fmapper
        self.orig = Mesh(template_v, faces, num_eigenvectors=self.fmapper.num_eigenvectors)
        self.processed = fmapper.preprocess_mesh(template_v, faces) if processed is None else processed
        self.exhaustive_search = exhaustive_search

        if symmetry_axis is None:
            self.mirror = None
        else:
            assert symmetry_axis in [0, 1, 2], 'Invalid symmetry axis'
            flip_v = template_v * [[-1 if i == symmetry_axis else 1 for i in range(3)]]
            bc, fidx = project_to_bccoord(template_v, faces, flip_v)
            self.mirror = bc2sparse(faces, fidx, bc, len(template_v))
            assert np.allclose(self.mirror @ template_v, flip_v, atol=0.03), 'Invalid mirror; not a perfect reflection'

        assert self.processed.v.shape[0] == fmapper.target_size, 'Number of vertices do not match config'
        assert self.processed.g.shape[0] == fmapper.target_size, 'Geodesic distances do not match config'
        assert self.processed.s.shape[1] == fmapper.sig_config.num_signatures, 'Signature functions do not match config'
        assert len(self.processed.eigen[0]) == fmapper.num_eigenvectors, 'Eigenvectors do not match config'
        assert self.processed.s.shape[1] == 121 and np.max(self.processed.s[:, -3:]) > 10, 'debugging'
        l2h, h2l = self.cartesian_mapping(self.processed.v, self.processed.f, template_v, faces)
        self.template_l2h, self.template_h2r = l2h, h2l

        self.logger = logging.getLogger('TemplateMapper')

    @staticmethod
    def cartesian_mapping(v1: np.ndarray, f1: np.ndarray, v2: np.ndarray, f2: np.ndarray) -> (coo_matrix, coo_matrix):
        """
        Compute the correspondence between two meshes using point-to-surface mapping in cartesian space
        return sparse maps between meshes (input meshes must be aligned!)
        """
        bc, fidx = project_to_bccoord(v1, f1, v2)
        map1_to_2 = bc2sparse(f1, fidx, bc, len(v1))

        bc, fidx = project_to_bccoord(v2, f2, v1)
        map2_to_1 = bc2sparse(f2, fidx, bc, len(v2))

        return map1_to_2, map2_to_1

    def align_to_template(self, verts, faces, query=None) -> (coo_matrix, coo_matrix):
        """
        Compute the correspondence to template mesh

        :param verts: np.ndarray
            Vertices of query mesh
        :param faces: np.ndarray
            Faces of query mesh
        :param query:
        -------
        query2template: coo_matrix
            sparse mapping from function defined on query vertices to template vertices
        template2query: coo_matrix
            sparse mapping from function defined on template vertices to query vertices
        """
        # low-res version of the mesh used for correspondence matching
        query_volume = mesh_volume(verts, faces)
        if query_volume < 0:
            self.logger.warning('Flipped normals detected in query mesh')
            faces = faces[:, [1, 0, 2]]
            query_volume *= -1

        query = self.fmapper.preprocess_mesh(verts, faces) if query is None else query
        assert all([np.size(e) > 0 for e in query.eigen]), 'Failed to compute eigenvectors'
        assert np.size(query.g) > 0, 'Failed to compute geodesic distances'
        assert np.size(query.s) > 0, 'Signature functions not set'
        assert query.s.shape[1] == 121 and np.max(query.s[:, -3:]) > 10, 'debugging'

        # compute the correspondence with default initial match dimensions
        i, j = self.fmapper.mesh_correspondence(self.processed, query)
        template_on_query = query.v[j]
        abs_volume = np.abs(mesh_volume(template_on_query, self.processed.f))
        dg = np.abs(query.g[j][:, j] - self.processed.g[i][:, i]).mean()
        success = (abs_volume / query_volume) * np.exp(-dg / 0.1) > 0.6
        deepfm = self.fmapper.resnet is not None

        # exhaustive search
        for dims in range(2, 7):
            if success: break
            self.logger.info(f'Exhaustive search with {dims} dimensions')

            for deepfm in [True, False]:
                if deepfm and self.fmapper.resnet is None:
                    self.logger.debug('Deep functional maps not available')
                    continue
                i, j = self.fmapper.mesh_correspondence(self.processed, query,
                                                        initial_dimensions=dims, use_deep_maps=deepfm)

                dg = np.abs(query.g[j][:, j] - self.processed.g[i][:, i]).mean()
                template_on_query = query.v[j]
                abs_volume = np.abs(mesh_volume(template_on_query, self.processed.f))

                success = (abs_volume / query_volume) * np.exp(-dg / 0.1) > 0.6
                if success: break

        # assert abs_volume > 0.95 * query_volume, 'Failed to align query to template'
        assert dg < 0.04, 'Failed to align query to template'

        # geodesic distortion
        self.logger.info(f'Geodesic distortion is {dg:.3f}' + ('*' if deepfm else ''))

        # smooth vertex positions but make sure they stay on the query manifold
        # for ii in range(5):
        # template_on_query = per_vertex_smoothing(template_on_query, self.processed.f)
        # template_on_query = igl.point_mesh_squared_distance(template_on_query, query.v, query.f)[2]

        # revert to original resolution
        template_mapped = self.template_l2h @ template_on_query

        # check if the match is flipped
        aligned_volume = mesh_volume(template_mapped, self.orig.f)
        if aligned_volume < 0 and self.mirror is None:
            self.logger.warning('Match found is flipped')
        elif aligned_volume < 0 and self.mirror is not None:
            self.logger.debug('Match found is flipped')
            template_mapped = self.mirror @ template_mapped

        template_mapped = per_vertex_smoothing(template_mapped, self.orig.f)

        # compute the sparse mappings
        query2template, template2query = self.cartesian_mapping(verts, faces, template_mapped, self.orig.f)

        return dg, query2template, template2query




