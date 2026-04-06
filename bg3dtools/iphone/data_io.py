"""
Stray Scanner iPhone data I/O utilities.

This module provides functions for reading and processing depth data captured
with the Stray Scanner iOS app, including camera intrinsics, poses, and
depth frame reconstruction.

Functions
---------
read_data
    Load Stray Scanner dataset from folder.
load_depth_img
    Load and optionally filter a single depth frame.
reconstruct_point_clouds
    Convert depth maps to point clouds with optional fusion.
"""

import os
from os.path import split, join
import numpy as np
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
from bg3dtools.render import get_heatmap_color
from bg3dtools.image_tools import vreader
from bg3dtools.pointclouds.reconstruction import depth_to_pc, scale_intrinsics

description = """
This script visualizes datasets collected using the Stray Scanner app.
"""

usage = """
Basic usage: python stray_visualize.py <path-to-dataset-folder>
"""

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
COLOR_WIDTH = 1920
COLOR_HEIGHT = 1440
MIN_DEPTH = 0.03
MAX_DEPTH = 20.0


def read_data(strayscanner_folder: str) -> dict:
    """
    Reads the Stray Scanner dataset from the specified path.
    Parameters
    ----------
    strayscanner_folder : str
        Path to the dataset folder.
    Returns
    -------
    dict
        A dictionary containing the camera intrinsics, poses, and depth frames.
        Keys: 'path', 'intrinsics', 'poses', 'depth_frames', 'mp4_file'.
    """
    intrinsics = np.loadtxt(os.path.join(strayscanner_folder, 'camera_matrix.csv'), delimiter=',')
    odometry = np.loadtxt(os.path.join(strayscanner_folder, 'odometry.csv'), delimiter=',', skiprows=1)
    timestamps = odometry[:, 0].tolist()
    positions = odometry[:, 2:5]       # (N, 3)
    quaternions = odometry[:, 5:]      # (N, 4)
    rotations = Rotation.from_quat(quaternions).as_matrix()  # (N, 3, 3)

    N = len(odometry)
    poses = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = positions

    fps = 1.0 / np.mean(np.diff(timestamps))
    depth_dir = os.path.join(strayscanner_folder, 'depth')

    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
    depth_frames = [f for f in depth_frames if '.npy' in f or '.png' in f]

    rgb_file = os.path.join(strayscanner_folder, 'rgb.mp4')
    assert os.path.isfile(rgb_file), 'missing expected video file %s' % rgb_file

    return { 'path': strayscanner_folder,
             'poses': poses,
             'intrinsics': intrinsics,
             'depth_frames': depth_frames,
             'mp4_file': rgb_file,
             'fps': fps,
             'source': 'strayscanner'}


def save_data(output_file, data, save_depth=True, depth_data=None):
    from bg3dtools.utils.cifs_wrappers import save_npz

    out_dict = {k: v for k, v in data.items()}  # copy dict

    if depth_data is not None:
        out_dict['depth_data'] = depth_data
    elif save_depth:
        out_dict['depth_data'] = load_depth(data['depth_frames'])

    save_npz(output_file, **out_dict)


def load_depth_img(depth_file: str, filter_level=0, scale=.001) -> np.ndarray:
    depth_path, file_name = split(depth_file)
    assert depth_path.endswith('depth')

    if depth_file.endswith('.npy'):
        depth_mm = np.load(depth_file)
    elif depth_file.endswith('.png'):
        depth_mm = np.array(imageio.imread(depth_file))
    else:
        raise ValueError(f"Unsupported depth file format: {depth_file}")

    depth_m = depth_mm.astype(np.float32) * scale

    if filter_level > 0:
        strayscanner_dir, _ = split(depth_path)
        confidence_file = join(strayscanner_dir, 'confidence', file_name)
        confidence = np.array(imageio.imread(confidence_file))
        depth_m[confidence < filter_level] = 0.0

    return depth_m


def load_depth(depth_frames: list, filter_level=0) -> np.ndarray:
    first = load_depth_img(depth_frames[0], filter_level=filter_level)
    depth_array = np.zeros((len(depth_frames), *first.shape), dtype=np.float32)
    depth_array[0] = first
    for i in range(1, len(depth_frames)):
        depth_array[i] = load_depth_img(depth_frames[i], filter_level=filter_level)

    return depth_array


def load_confidence(confidence_file: str) -> np.ndarray:
    return np.array(imageio.imread(confidence_file), dtype=np.int32)


def trajectory(data):
    """
    Returns a single LineSet connecting each camera pose's world frame position.
    returns: [open3d.geometry.LineSet]
    """
    import open3d as o3d
    positions = data['poses'][:, :3, 3]  # (N, 3)
    N = len(positions)
    if N < 2:
        return []
    lines = np.column_stack([np.arange(N - 1), np.arange(1, N)])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return [line_set]


def show_frames(data, every=15):
    """
    Returns a list of meshes of coordinate axes that have been transformed to represent the camera matrix
    at each --every:th frame.

    data: dict with keys ['poses', 'intrinsics']
    returns: [open3d.geometry.TriangleMesh]
    """
    import open3d as o3d
    frames = [o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.25, np.zeros(3))]
    for i, T_WC in enumerate(data['poses']):
        if not i % every == 0:
            continue
        print(f"Frame {i}", end="\r")
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, np.zeros(3))
        frames.append(mesh.transform(T_WC))
    return frames


def reconstruct_point_clouds(data: dict, *,
                             return_as_numpy=False,
                             globalize=False,
                             every=1,
                             filter_level=1,
                             compute_normals=True,
                             remove_edges=True):
    """
    Converts depth maps to point clouds and merges them all into one global point cloud.
    data: dict with keys ['path', 'intrinsics', 'poses']
    returns: [open3d.geometry.PointCloud]
    """
    path = data['path']
    point_clouds = []
    depth_list = []
    color_intrinsics = data['intrinsics']
    depth_intrinsics = scale_intrinsics(color_intrinsics,
                                        (COLOR_WIDTH, COLOR_HEIGHT),
                                        (DEPTH_WIDTH, DEPTH_HEIGHT))

    rgb_path = os.path.join(path, 'rgb.mp4')
    video = vreader(rgb_path)
    for i, (T_WC, rgb) in enumerate(zip(data['poses'], video)):
        if i % every != 0:
            continue
        print(f"Point cloud {i}", end="\r")

        depth_path = data['depth_frames'][i]
        depth = load_depth_img(depth_path, filter_level=filter_level)
        depth_list.append(depth)

        pc = depth_to_pc(depth, depth_intrinsics, rgb=rgb,
                         compute_normals=compute_normals,
                         remove_edges=remove_edges)
        if globalize:
            pc.transform(T_WC)

        point_clouds.append(pc)

    if return_as_numpy:
        points = [np.asarray(pc.points) for pc in point_clouds]
        colors = [np.asarray(pc.colors) for pc in point_clouds]
        normals = [np.asarray(pc.normals) for pc in point_clouds] if compute_normals else None
        point_clouds = (points, colors, normals)

    return point_clouds, depth_list


def load_and_show(flags):
    import open3d as o3d
    data = read_data(flags.path)
    point_clouds = reconstruct_point_clouds(data, globalize=flags.globalize,
                                            compute_normals=flags.compute_normals,
                                            remove_edges=flags.remove_edges)

    geometries = point_clouds
    if flags.globalize:
        camera_trajectories = trajectory(data)
        camera_axes = show_frames(data, 15)
        geometries += camera_trajectories + camera_axes
    # visualize
    o3d.visualization.draw_geometries(geometries, window_name='Stray Scanner Dataset Visualization')


def add_arguments(parser):
    parser.add_argument('--path', type=str, default="", help="Path to StrayScanner dataset to process.")
    parser.add_argument('--globalize', action='store_true', help="Apply extrinsic transformation to put all frames in shared coordinate space.")
    parser.add_argument('--every', type=int, default=4, help="Show only every nth point cloud and coordinate frames. Only used for point cloud and odometry visualization.")
    parser.add_argument('--confidence', '-c', type=int, default=1, help="Keep only depth estimates with confidence equal or higher to the given value. There are three different levels: 0, 1 and 2. Higher is more confident.")
    parser.add_argument('--compute_normals', action='store_true', help="Compute normals from depth maps.")
    parser.add_argument('--remove_edges', action='store_true', help="Remove depth edges from point clouds.")
    return parser


def pcds_to_video(pcd_list, depths,
                  outfile="pointclouds.mp4",
                  width=720, height=720, fps=60,
                  eye=(0, 0, 0),      # eye
                  lookat=(0, 0, 1),    # look-at
                  up=(0, 1, 0)):       # typically Y-up
    import matplotlib.pyplot as plt
    import open3d as o3d
    from PIL import Image

    # 1 — prepare an off-screen renderer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.get_render_option().background_color = np.array([1, 1, 1])

    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    front = np.subtract(eye, lookat)
    ctr.set_front(front / np.linalg.norm(front))
    ctr.set_up(up)

    # 2 — open a video writer
    with imageio.get_writer(outfile, fps=fps, codec="libx264",
                            quality=8, macro_block_size=None) as vid:
        for idx, (pcd, depth) in enumerate(zip(pcd_list, depths)):
            vis.clear_geometries()
            vis.add_geometry(pcd)
            ctr.set_lookat(lookat)
            front = np.subtract(eye, lookat)
            ctr.set_front(front / np.linalg.norm(front))
            ctr.set_up(up)
            ctr.set_zoom(0.265)
            vis.poll_events()
            vis.update_renderer()
            pc_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))

            d_img = Image.fromarray(depth)
            d_img = d_img.rotate(270)
            d_img = d_img.resize((width, height), Image.BILINEAR)  # Resize depth image to match video size
            d_frame = get_heatmap_color(np.asarray(d_img), caxis=(0.8, 1.5), mapname='parula')
            frame = np.column_stack([pc_frame, d_frame])
            vid.append_data((frame * 255).astype(np.uint8))
            if idx == 0:
                fig, ax = plt.subplots()
                im = ax.imshow(frame)
            else:
                im.set_data(frame)
            plt.pause(0.2)

    plt.close('all')
    vis.destroy_window()
    print(f"Saved {len(pcd_list)} frames → {outfile}")


