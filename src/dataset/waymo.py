# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import imageio
import torch
from torch.utils.data import IterableDataset
import glob
import sys
sys.path.append('../')
from .data_utils import get_nearby_view_ids, loader_resize, random_crop, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses
from .shims.crop_shim import apply_crop_shim

from dataclasses import dataclass
from typing import Literal
from pathlib import Path
from .dataset import DatasetCfgCommon
    
import random
from .base_utils import downsample_gaussian_blur
    

class WaymoStaticDataset(IterableDataset):
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    
    def __init__(self, args, mode, step_tacker, **kwargs):
        self.folder_path = os.path.join('data/waymo/processed/training')
        self.dataset_name = 'waymo'
        self.pose_noise_level = 0

        self.args = args
        self.mode = mode  # train / test / validation
        # self.num_source_views = args.view_sampler.num_context_views
        self.num_source_views = 5
        self.llffhold = 8
        self.random_crop = False
        self.rectify_inplane_rotation = False
        
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        # self.image_size = (176, 240)
        self.image_size = (352, 480)

        all_scenes = os.listdir(self.folder_path)
        
        ############     Wamyo Parameters     ############
        self.num_cams, self.camera_list =1, [0]
        self.start_timestep, self.end_timestep = 0, 198
        
        if len(args.scene) > 0:
            self.scenes = args.scene
        else:
            self.scenes = all_scenes

        print("loading {} for {}".format(self.scenes, mode))
        print(f'[INFO] num scenes: {len(self.scenes)}')
        for i, scene in enumerate(self.scenes):
            scene_path = os.path.join(self.folder_path, scene)
            
            rgb_files = []
            for t in range(self.start_timestep, self.end_timestep):
                for cam_idx in self.camera_list:
                    rgb_files.append(os.path.join(scene_path, "images", f"{t:03d}_{cam_idx}.jpg"))
            
            intrinsics, c2w_mats = self.load_calibrations(scene_path)
            
            
            near_depth = 0.1
            far_depth = 100.0
            
            i_test = np.arange(len(rgb_files))[::self.llffhold] if mode != 'eval_pose' else []
            i_train = np.array([j for j in np.arange(len(rgb_files)) if
                                (j not in i_test and j not in i_test)])

            if mode == 'train':
                i_render = i_train
            else:
                i_render = i_test

            
            self.train_intrinsics.append([intrinsics] * len(i_train))
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics] * num_render)
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)


    def load_calibrations(self, scene_path):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, ego-to-world matrices, etc.
        """
        # to store per-camera intrinsics and extrinsics
        _intrinsics = []
        cam_to_egos = []
        for i in range(self.num_cams):
            # load camera intrinsics
            # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}].
            # ====!! we did not use distortion parameters for simplicity !!====
            # to be improved!!
            intrinsic = np.loadtxt(
                os.path.join(scene_path, "intrinsics", f"{i}.txt")
            )
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            # scale intrinsics w.r.t. load size
            fx, fy = (
                fx * self.image_size[1] / self.ORIGINAL_SIZE[i][1],
                fy * self.image_size[0] / self.ORIGINAL_SIZE[i][0],
            )
            cx, cy = (
                cx * self.image_size[1] / self.ORIGINAL_SIZE[i][1],
                cy * self.image_size[0] / self.ORIGINAL_SIZE[i][0],
            )
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            _intrinsics.append(intrinsic)

            # load camera extrinsics
            cam_to_ego = np.loadtxt(
                os.path.join(scene_path, "extrinsics", f"{i}.txt")
            )
            # because we use opencv coordinate system to generate camera rays,
            # we need a transformation matrix to covnert rays from opencv coordinate
            # system to waymo coordinate system.
            # opencv coordinate system: x right, y down, z front
            # waymo coordinate system: x front, y left, z up
            cam_to_egos.append(cam_to_ego @ self.OPENCV2DATASET)

        # compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, cam_ids = [], []
        # ===! for waymo, we simplify timestamps as the time indices
        timestamps, timesteps = [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(
            os.path.join(scene_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(
                os.path.join(scene_path, "ego_pose", f"{t:03d}.txt")
            )
            # compute ego_to_world transformation
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            ego_to_worlds.append(ego_to_world)
            for cam_id in self.camera_list:
                cam_ids.append(cam_id)
                # transformation:
                #   (opencv_cam -> waymo_cam -> waymo_ego_vehicle) -> current_world
                cam2world = ego_to_world @ cam_to_egos[cam_id]
                cam_to_worlds.append(cam2world)
                intrinsics.append(_intrinsics[cam_id])
                # ===! we use time indices as the timestamp for waymo dataset for simplicity
                # ===! we can use the actual timestamps if needed
                # to be improved
                timestamps.append(t - self.start_timestep)
                timesteps.append(t - self.start_timestep)

        intrinsics = np.stack(intrinsics, axis=0)
        cam_to_worlds = np.stack(cam_to_worlds, axis=0)
        return intrinsic, cam_to_worlds



    def get_data_one_batch(self, idx, nearby_view_id=None):
        self.nearby_view_id = nearby_view_id
        return self.__getitem__(idx=idx)

    def num_poses(self):
        return len(self.render_rgb_files)

    def __len__(self):
        return len(self.render_rgb_files)

    def __iter__(self):
        for idx in range(len(self.render_rgb_files)):
            rgb_file = self.render_rgb_files[idx]
            scene, img_idx = rgb_file.split("/")[4], rgb_file[-7:-4]
            rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
            render_pose = self.render_poses[idx]
            intrinsics = self.render_intrinsics[idx]
            depth_range = self.render_depth_range[idx]

            train_set_id = self.render_train_set_ids[idx]
            train_rgb_files = self.train_rgb_files[train_set_id]
            train_poses = self.train_poses[train_set_id]
            train_intrinsics = self.train_intrinsics[train_set_id]

            img_size = rgb.shape[:2]
            camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                    render_pose.flatten())).astype(np.float32)

            if self.mode == 'train':
                id_render = train_rgb_files.index(rgb_file)
                subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
                # num_select = self.num_source_views + np.random.randint(low=-2, high=3)
                num_select = self.num_source_views
                
            else:
                id_render = -1
                subsample_factor = 1
                num_select = self.num_source_views

            nearest_pose_ids = None
            # num_select = min(self.num_source_views*subsample_factor, 22)
            nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                    train_poses,
                                                    num_select=num_select,
                                                    tar_id=id_render,
                                                    angular_dist_method='dist')

            nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

            assert id_render not in nearest_pose_ids
            # occasionally include input image
            if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
                nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

            src_rgbs = []
            src_cameras = []
            src_intrinsics, src_extrinsics = [], []
            for id in nearest_pose_ids:
                src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
                train_pose = train_poses[id]
                train_intrinsics_ = train_intrinsics[id]
                src_extrinsics.append(train_pose)
                src_intrinsics.append(train_intrinsics_)
                if self.rectify_inplane_rotation:
                    train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

                src_rgbs.append(src_rgb)
                img_size = src_rgb.shape[:2]
                src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                            train_pose.flatten())).astype(np.float32)
                src_cameras.append(src_camera)

            src_rgbs = np.stack(src_rgbs, axis=0)
            src_cameras = np.stack(src_cameras, axis=0)
            src_intrinsics, src_extrinsics = np.stack(src_intrinsics, axis=0), np.stack(src_extrinsics, axis=0)
            # if self.mode == 'train' and random.randint(1, 100) < 75:
            #     rgb, camera, src_rgbs, src_cameras, intrinsics, src_intrinsics = random_crop(rgb,camera,src_rgbs,src_cameras, size=self.image_size, center=None)
            # elif self.mode == 'train':
            #     rgb, camera, src_rgbs, src_cameras, intrinsics, src_intrinsics = loader_resize(rgb,camera,src_rgbs,src_cameras, size=self.image_size)
            rgb, _, src_rgbs, _, _, _ = loader_resize(rgb,camera,src_rgbs,src_cameras, size=self.image_size)
            
            src_extrinsics = torch.from_numpy(src_extrinsics).float()
            extrinsics = torch.from_numpy(render_pose).unsqueeze(0).float()
            
            src_intrinsics = self.normalize_intrinsics(torch.from_numpy(src_intrinsics[:,:3,:3]).float(), self.image_size)
            intrinsics = self.normalize_intrinsics(torch.from_numpy(intrinsics[:3,:3]).unsqueeze(0).float(), self.image_size)

            depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

            # Resize the world to make the baseline 1.
            if src_extrinsics.shape[0] == 2:
                a, b = src_extrinsics[:, :3, 3]
                scale = (a - b).norm()
                if scale < 0.001:
                    print(
                        f"Skipped {scene} because of insufficient baseline "
                        f"{scale:.6f}"
                    )
                src_extrinsics[:, :3, 3] /= scale
                extrinsics[:, :3, 3] /= scale
            else:
                scale = 1
                
            example = {
                    "context": {
                            "extrinsics": src_extrinsics,
                            "intrinsics": src_intrinsics,
                            "image": torch.from_numpy(src_rgbs[..., :3]).permute(0, 3, 1, 2),
                            "near":  (depth_range[0].repeat(num_select) / scale).float(),
                            "far": (depth_range[1].repeat(num_select) / scale).float(),
                            "index": torch.from_numpy(nearest_pose_ids),
                    },
                    "target": {
                            "extrinsics": extrinsics,
                            "intrinsics": intrinsics,
                            "image": torch.from_numpy(rgb[..., :3]).unsqueeze(0).permute(0, 3, 1, 2),
                            "near": (depth_range[0].unsqueeze(0) / scale).float(),
                            "far": (depth_range[1].unsqueeze(0) / scale).float(),
                            "index": torch.tensor([int(img_idx)]),
                    
                    },"scene":scene}
            yield apply_crop_shim(example, tuple(self.image_size))
        
    def normalize_intrinsics(self, intrinsics, img_size):
        h, w = img_size
        # 归一化内参矩阵
        intrinsics_normalized = intrinsics.clone()
        intrinsics_normalized[:, 0, 0] /= w
        intrinsics_normalized[:, 1, 1] /= h
        intrinsics_normalized[:, 0, 2] = 0.5
        intrinsics_normalized[:, 1, 2] = 0.5
        return intrinsics_normalized