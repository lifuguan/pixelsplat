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
from turtle import width
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
    

class LLFFTestDataset(IterableDataset):
    def __init__(self, args, mode, step_tacker, **kwargs):
        self.folder_path = os.path.join('data/ibrnet/eval', 'nerf_llff_data/')
        self.dataset_name = 'llff'
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
        self.idx_to_node_id_list = []
        self.node_id_to_idx_list = []
        self.train_view_graphs = []

        # self.image_size = (352, 480) #测试的时候
        self.image_size = (320, 448) #测试的时候
        out_w = 448
        self.ratio = out_w / 504
        self.h, self.w = int(self.ratio*378), int(out_w)
        
        all_scenes = os.listdir(self.folder_path)
        
        
        if len(args.scene) > 0:
            self.scenes = args.scene
        else:
            self.scenes = all_scenes

        print("loading {} for {}".format(self.scenes, mode))
        print(f'[INFO] num scenes: {len(self.scenes)}')
        for i, scene in enumerate(self.scenes):
            scene_path = os.path.join(self.folder_path, scene)
            factor = 8 #4
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene_path, load_imgs=False, factor=factor)
            
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)
            
            i_test = np.arange(poses.shape[0])[::self.llffhold] if mode != 'eval_pose' else []
            i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                (j not in i_test and j not in i_test)])

            # pose_initializer = PoseInitializer(
            #     data_path=scene_path,
            #     image_ids=i_train,
            #     load_external=True,
            #     args=args
            # )
            # self.train_view_graphs.append(pose_initializer.view_graph)
            
            idx_to_node_id, node_id_to_idx = {}, {}
            for j in range(i_train.shape[0]):
                idx_to_node_id[j] = i_train[j]
                node_id_to_idx[i_train[j]] = j
            self.idx_to_node_id_list.append(idx_to_node_id)
            self.node_id_to_idx_list.append(node_id_to_idx)

            if mode == 'train':
                i_render = i_train
            else:
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)

    def get_data_one_batch(self, idx, nearby_view_id=None):
        self.nearby_view_id = nearby_view_id
        return self.__getitem__(idx=idx)

    def num_poses(self):
        return len(self.render_rgb_files)

    def __len__(self):
        return len(self.render_rgb_files)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]
    def __iter__(self):
        self.shuffle(self.render_rgb_files)
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
            # view_graph = self.train_view_graphs[train_set_id]
            idx_to_node_id = self.idx_to_node_id_list[train_set_id]
            node_id_to_idx = self.node_id_to_idx_list[train_set_id]

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
            if self.mode == 'train' and random.randint(1, 100) < 0:
                #(168, 224)#(378,504)
                # low=168*224//504+2
                # high=370
                # crop_h = np.random.randint(100, 300) #74~378
                # crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
                # crop_w = int(168*224 / crop_h)
                # crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
                # self.image_size=(crop_h,crop_w)
                rgb, camera, src_rgbs, src_cameras, intrinsics, src_intrinsics,center_h,center_w = random_crop(rgb,camera,src_rgbs,src_cameras, size=self.image_size, center=None)
                src_intrinsics = self.normalize_intrinsics_1(torch.from_numpy(src_intrinsics[:,:3,:3]).float(), self.image_size,center_h,center_w)
                intrinsics = self.normalize_intrinsics_1(torch.from_numpy(intrinsics[:3,:3]).unsqueeze(0).float(), self.image_size,center_h,center_w)
                # print("crop")
            elif self.mode == 'train':
                rgb, camera, src_rgbs, src_cameras, intrinsics, src_intrinsics = loader_resize(rgb,camera,src_rgbs,src_cameras, size=self.image_size)
                center_h=0.5
                center_w=0.5
                src_intrinsics = self.normalize_intrinsics(torch.from_numpy(src_intrinsics[:,:3,:3]).float(), self.image_size,center_h,center_w)
                intrinsics = self.normalize_intrinsics(torch.from_numpy(intrinsics[:3,:3]).unsqueeze(0).float(), self.image_size,center_h,center_w)
                # print("resize")
            else:   #测试
                rgb, camera, src_rgbs, src_cameras, intrinsics, src_intrinsics = loader_resize(rgb,camera,src_rgbs,src_cameras, size=self.image_size)
                center_h=0.5
                center_w=0.5
                src_intrinsics = self.normalize_intrinsics(torch.from_numpy(src_intrinsics[:,:3,:3]).float(), self.image_size,center_h,center_w)
                intrinsics = self.normalize_intrinsics(torch.from_numpy(intrinsics[:3,:3]).unsqueeze(0).float(), self.image_size,center_h,center_w)
              

            extrinsics = torch.from_numpy(render_pose).unsqueeze(0).float()
            src_extrinsics = torch.from_numpy(src_extrinsics).float()
            depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

            # Resize the world to make the baseline 1.
            if src_extrinsics.shape[0] == 1:
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
        
    def normalize_intrinsics(self, intrinsics, img_size,center_h,center_w):
        h, w = img_size
        # 归一化内参矩阵
        intrinsics_normalized = intrinsics.clone()
        intrinsics_normalized[:, 0, 0] /= w
        intrinsics_normalized[:, 1, 1] /= h
        intrinsics_normalized[:, 0, 2] = 0.5
        intrinsics_normalized[:, 1, 2] = 0.5
        return intrinsics_normalized
    def normalize_intrinsics_1(self, intrinsics, img_size,center_h,center_w):
        h, w = img_size
        # 归一化内参矩阵
        intrinsics_normalized = intrinsics.clone().float()
        intrinsics_normalized[:, 0, 0] /= w
        intrinsics_normalized[:, 1, 1] /= h
        intrinsics_normalized[:, 0, 2] /= w
        intrinsics_normalized[:, 1, 2] /= h
        return intrinsics_normalized