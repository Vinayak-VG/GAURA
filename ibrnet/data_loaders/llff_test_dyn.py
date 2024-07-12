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
import sys

sys.path.append("../")
from torch.utils.data import Dataset
from .data_utils import random_crop, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses
import random
from ibrnet.data_loaders.create_haze.create_hazy import gen_haze
from ibrnet.data_loaders.create_blur.gen_blur import motion_gen_blur, defocus_gen_blur
from ibrnet.data_loaders.create_dark.create_dark import process_image
from ibrnet.data_loaders.create_rain.create_rain import create_rain
from ibrnet.data_loaders.create_noise.create_noise import burst_noise
from ibrnet.data_loaders.create_snow.create_snow import snow_sim

class LLFFTestDatasetDynamic(Dataset):
    
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
            
        self.folder_path = os.path.join("/home/vinayak/restoration_nerf_finalsprint/data/nerf_llff_data/")
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        all_scenes = os.listdir(self.folder_path)
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(
                scene_path, load_imgs=False, factor=4
            )
            
            near_depth = np.min(bds)
            far_depth = np.max(bds)

            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            i_test = np.arange(poses.shape[0])[:: self.args.llffhold]
            i_train = np.array(
                [
                    j
                    for j in np.arange(int(poses.shape[0]))
                    if (j not in i_test and j not in i_test)
                ]
            )

            if mode == "train":
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
            self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
            self.render_train_set_ids.extend([i] * num_render)

    def __len__(self):
        return (
            len(self.render_rgb_files) * 100000
            if self.mode == "train"
            else len(self.render_rgb_files)
        )

    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)
        if self.args.ft_corrup == "gen":
            corruptions = [
                           ("dark", 0), 
                           ("haze", 1),  
                           ("rain", 2), 
                           ("motion", 3), 
                           ("burstnoise", 4),  
                           ("clean", 5),
                           ("snow", 6)
                        #    ("defocus", 5)
                        #    ("underwater", 6), 
                        ]
            corruption, embed_id1 = random.choice(corruptions)
        elif self.args.ft_corrup == "threecorr":
            corruptions = [
                           ("dark", 0), 
                           ("haze", 1),  
                           ("motion", 2),  
                        ]
            corruption, embed_id1 = random.choice(corruptions)
        else:
            corruption = self.args.ft_corrup
            embed_id1 = 6
        
        rgb_file_ = self.render_rgb_files[idx]
        rgb_file = os.path.join(os.path.dirname(rgb_file_) + "", os.path.basename(rgb_file_))
        
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        if self.mode == "train":
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file_)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=2)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            min(self.num_source_views * subsample_factor, 28),
            tar_id=id_render,
            angular_dist_method="dist",
        )
        nearest_pose_ids = np.random.choice(
            nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False
        )

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        if corruption == "haze":
            beta = random.random() * 4 + 2
            beta = beta/2.
            A_light = random.randint(125, 200)
            target = beta
        elif corruption == "dark":
            scale = random.randint(6, 30)
            noise_prob = random.randint(0, 1)
            target = scale
        elif corruption == "rain":
            theta = np.random.randint(60, 120)
            density = random.randint(-7, -3)
            target = density
        elif corruption == "burstnoise":
            gain = random.randint(12, 24)
            target = gain
        elif corruption == "snow":
            angle = random.randrange(-30, 30, 5)
            speed = random.choice([0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05])
            flake_size = random.random() * 0.5 + 0.3
            density_uniform = random.choice([0.5, 0.6, 0.7, 0.8])
            flake_size_uniformity = random.choice([0.4,0.5,0.6,0.8])
            density = random.random() * 0.05 + 0.04
            target = density

        for id in nearest_pose_ids:
            src_rgb_file = train_rgb_files[id]
            src_rgb_file_depth = os.path.join(os.path.dirname(src_rgb_file) + "_depth", os.path.basename(src_rgb_file))
            src_rgb_file = os.path.join(os.path.dirname(src_rgb_file) + "", os.path.basename(src_rgb_file))
            if corruption == "haze":
                src_rgb = (gen_haze(src_rgb_file, src_rgb_file_depth, beta, A_light) / 255.0).astype(np.float32)
            elif corruption == "clean":
                src_rgb = imageio.imread(src_rgb_file).astype(np.float32) / 255.0
                target = 0
            elif corruption == "motion":
                size = np.random.randint(5, 26)
                angle_list = list(range(0, 10)) + list(range(80, 100)) + list(range(170, 180))
                angle = random.choice(angle_list)
                target = size
                src_rgb = (motion_gen_blur(src_rgb_file, size, angle, corruption) / 255.0).astype(np.float32)
            elif corruption == "defocus":
                blur_amount = np.random.randint(5, 11)
                depth_list = [-1, 1]
                depth_type = random.choice(depth_list)
                src_rgb = (defocus_gen_blur(src_rgb_file, src_rgb_file_depth, blur_amount, depth_type) / 255.0).astype(np.float32)
            elif corruption == "dark":
                src_rgb = np.float32(process_image(src_rgb_file, scale, noise_prob))
            elif corruption == "rain":
                src_rgb = create_rain(src_rgb_file, theta=theta, density=density)
            elif corruption == "burstnoise":
                src_rgb = np.float32(burst_noise(src_rgb_file, gain = gain))
            elif corruption == "snow":
                src_rgb = np.float32(snow_sim(src_rgb_file, density, density_uniform, flake_size, flake_size_uniformity, angle, speed))
            
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        # clean_rgbs = []
        # for id in nearest_pose_ids:
        #     clean_rgb_file = train_rgb_files[id]
        #     clean_rgb_file = os.path.join(os.path.dirname(clean_rgb_file), os.path.basename(clean_rgb_file))
        #     clean_rgb = imageio.imread(clean_rgb_file).astype(np.float32) / 255.0
        #     clean_rgbs.append(clean_rgb)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        # clean_rgbs = np.stack(clean_rgbs, axis=0)

        if self.mode == "train" and self.random_crop:
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(
                rgb, camera, src_rgbs, src_cameras, (crop_h, crop_w)
            )

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])

        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            # "clean_rgbs": torch.from_numpy(clean_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
            "embed_id1": torch.LongTensor([embed_id1]),
            "target": torch.LongTensor([target])
        }