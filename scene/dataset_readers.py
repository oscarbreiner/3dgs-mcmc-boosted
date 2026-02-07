#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    random_ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def _sample_random_points(nerf_normalization, num_pts, inside_out=False):
    radius = float(nerf_normalization["radius"])
    if not inside_out:
        return np.random.random((num_pts, 3)) * (radius * 6.0) - (radius * 3.0)

    # Inside-out adaptation: sample inside a sphere centered at camera centroid.
    center = -np.asarray(nerf_normalization["translate"], dtype=np.float32)
    dirs = np.random.normal(size=(num_pts, 3)).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    dirs = dirs / norms
    radii = (np.random.random((num_pts, 1)).astype(np.float32) ** (1.0 / 3.0)) * radius
    return center[None, :] + dirs * radii

def readColmapSceneInfo(path, images, eval, llffhold=8, init_type="sfm", num_pts=100000, random_init_inside_out=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    pcd = None
    if init_type == "sfm":
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    elif init_type == "random":
        run_tag = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        ply_path = os.path.join(path, "random_{}.ply".format(run_tag))
        print(f"Generating random point cloud ({num_pts})...")
        if random_init_inside_out:
            print("Using inside-out random init adaptation.")
        xyz = _sample_random_points(
            nerf_normalization,
            num_pts=num_pts,
            inside_out=random_init_inside_out,
        )
        
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        print("Please specify a correct init_type: random or sfm")
        exit(0)

    if pcd is None:
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    random_ply_path = ply_path if init_type == "random" else None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           random_ply_path=random_ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def _nerfstudio_fov_from_intrinsics(fl_x, fl_y, w, h):
    fovx = 2 * np.arctan(w / (2 * fl_x))
    fovy = 2 * np.arctan(h / (2 * fl_y))
    return fovx, fovy

def _load_image_with_mask(image_path, mask_path, white_background):
    image = Image.open(image_path)
    if mask_path is None or not os.path.exists(mask_path):
        return image
    mask = Image.open(mask_path).convert("L")
    im_data = np.array(image.convert("RGB"), dtype=np.float32)
    mask_data = np.array(mask, dtype=np.float32) / 255.0
    mask_data = np.expand_dims(mask_data, axis=-1)
    bg = np.array([1, 1, 1], dtype=np.float32) if white_background else np.array([0, 0, 0], dtype=np.float32)
    im_data = im_data * mask_data + (1.0 - mask_data) * bg * 255.0
    return Image.fromarray(im_data.astype(np.uint8), "RGB")

def readNerfstudioSceneInfo(path, transforms_path, images_dir, white_background, eval):
    cam_infos = []
    with open(transforms_path) as json_file:
        contents = json.load(json_file)

    w = int(contents["w"])
    h = int(contents["h"])
    fl_x = float(contents["fl_x"])
    fl_y = float(contents["fl_y"])
    fovx, fovy = _nerfstudio_fov_from_intrinsics(fl_x, fl_y, w, h)

    frames = contents.get("frames", [])
    test_frames = contents.get("test_frames", [])
    if not eval:
        frames = frames + test_frames
        test_frames = []

    def _to_cam_infos(frames_list):
        infos = []
        for idx, frame in enumerate(frames_list):
            cam_rel = frame["file_path"]
            image_path = os.path.join(images_dir, cam_rel)
            if not os.path.exists(image_path):
                alt_path = os.path.join(path, cam_rel)
                if os.path.exists(alt_path):
                    image_path = alt_path
            mask_path = None
            if "mask_path" in frame:
                mask_path = os.path.join(images_dir, frame["mask_path"])
                if not os.path.exists(mask_path):
                    alt_mask = os.path.join(path, frame["mask_path"])
                    if os.path.exists(alt_mask):
                        mask_path = alt_mask
            image = _load_image_with_mask(image_path, mask_path, white_background)

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image_name = Path(cam_rel).stem
            infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image,
                                    image_path=image_path, image_name=image_name, width=w, height=h))
        return infos

    train_cam_infos = _to_cam_infos(frames)
    test_cam_infos = _to_cam_infos(test_frames)

    nerf_normalization = getNerfppNorm(train_cam_infos) if train_cam_infos else getNerfppNorm(test_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None,
                           random_ply_path=None)
    return scene_info

def readScanNetPPSceneInfo(path, images, eval, white_background, init_type="random", num_pts=100000, random_init_inside_out=False):
    images = images or "images"
    images = images.strip()

    dslr_transforms = os.path.join(path, "dslr", "nerfstudio", "transforms_undistorted.json")
    dslr_transforms_default = os.path.join(path, "dslr", "nerfstudio", "transforms.json")
    iphone_transforms = os.path.join(path, "iphone", "nerfstudio", "transforms.json")

    use_iphone = images.startswith("iphone/")
    if use_iphone and os.path.exists(iphone_transforms):
        transforms_path = iphone_transforms
    elif os.path.exists(dslr_transforms):
        transforms_path = dslr_transforms
    else:
        transforms_path = dslr_transforms_default

    if images == "images":
        if os.path.exists(os.path.join(path, "dslr", "resized_undistorted_images")):
            images = "dslr/resized_undistorted_images"
        elif os.path.exists(os.path.join(path, "dslr", "resized_images")):
            images = "dslr/resized_images"
        elif os.path.exists(os.path.join(path, "iphone", "rgb")):
            images = "iphone/rgb"

    images_dir = os.path.join(path, images)
    scene_info = readNerfstudioSceneInfo(path, transforms_path, images_dir, white_background, eval)

    pcd = None
    ply_path = None
    random_ply_path = None

    if init_type == "pc_aligned":
        ply_path = os.path.join(path, "scans", "pc_aligned.ply")
        if os.path.exists(ply_path):
            pcd = fetchPly(ply_path)
    elif init_type == "sfm":
        colmap_dir = os.path.join(path, "dslr", "colmap")
        ply_path = os.path.join(colmap_dir, "points3D.ply")
        bin_path = os.path.join(colmap_dir, "points3D.bin")
        txt_path = os.path.join(colmap_dir, "points3D.txt")
        if not os.path.exists(ply_path):
            if os.path.exists(bin_path):
                xyz, rgb, _ = read_points3D_binary(bin_path)
                storePly(ply_path, xyz, rgb)
            elif os.path.exists(txt_path):
                xyz, rgb, _ = read_points3D_text(txt_path)
                storePly(ply_path, xyz, rgb)
        if os.path.exists(ply_path):
            pcd = fetchPly(ply_path)
    elif init_type == "random":
        run_tag = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        ply_path = os.path.join(path, "random_{}.ply".format(run_tag))
        print(f"Generating random point cloud ({num_pts})...")
        if random_init_inside_out:
            print("Using inside-out random init adaptation.")
        xyz = _sample_random_points(
            scene_info.nerf_normalization,
            num_pts=num_pts,
            inside_out=random_init_inside_out,
        )
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        random_ply_path = ply_path
    else:
        print("Please specify a correct init_type: random, sfm, or pc_aligned")
        exit(0)

    if pcd is None or ply_path is None or not os.path.exists(ply_path):
        raise FileNotFoundError(
            "ScanNetPP init_type '{}' did not produce a valid point cloud. "
            "Check that the expected data exists under {}".format(init_type, path)
        )

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=scene_info.train_cameras,
                           test_cameras=scene_info.test_cameras,
                           nerf_normalization=scene_info.nerf_normalization,
                           ply_path=ply_path,
                           random_ply_path=random_ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "ScanNetPP": readScanNetPPSceneInfo
}
