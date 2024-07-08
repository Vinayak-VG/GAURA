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
from torch.utils.data import Dataset
import sys
import json
from glob import glob

sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids

def read_cameras(scene_path, pose_file):
    with open(pose_file, "r") as fp:
        meta = json.load(fp)
    
    intrinsics = np.eye(4)
    intrinsics_ = meta["intrinsics"]
    intrinsics[:3, :3] = intrinsics_
    rgb_files = []
    c2w_mats = []

    for i, frame in enumerate(meta["c2ws"]):
        rgb_file = os.path.join(scene_path, frame + ".png")
        rgb_files.append(rgb_file)
        c2w = np.array(meta["c2ws"][frame])
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats.append(c2w_opencv)
        if len(frame.split("_")) != 2:
            rgb_file = os.path.join(scene_path, frame + "_gt.png")
            rgb_files.append(rgb_file)
            c2w_mats.append(c2w_opencv)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics] * len(rgb_files)), c2w_mats


class ObjaverseZero12345Dataset(Dataset):
    def __init__(
        self,
        args,
        mode,
        scenes=(),
        **kwargs
    ):
        self.folder_path = "/objaverse-processed/zero12345_img"
        # /objaverse-processed/zero12345_img/zero12345_wide
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        if mode == "validation":
            mode = "val"
        assert mode in ["train", "val", "test"]
        self.mode = mode  # train / test / val
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip

        all_cats = os.listdir(os.path.join(self.folder_path, "zero12345_narrow"))
        self.render_rgb_files = []
        self.render_poses = []
        self.render_intrinsics = []
        self.render_train_set_ids = []
        self.train_rgb_files = []
        self.train_poses = []
        self.train_intrinsics = [] 

        cntr = 0
        for cat in all_cats:
            cat_path = os.path.join(self.folder_path, "zero12345_narrow", cat)
            scenes = os.listdir(cat_path)
            for scene in scenes:
                self.scene_path = os.path.join(self.folder_path, "zero12345_narrow", cat, scene)
                # if "000-028/12764f6bee6d4d64820ec117b791d6dc" in self.scene_path: continue
                # scene_files = glob(os.path.join(self.scene_path, "*.png"))
                # scene_files = [scene_f for scene_f in scene_files if "depth" not in scene_f]
                # if len(scene_files) == 40:
                #     pass
                # else:
                #     continue
                rgb_files, intrinsics, poses = read_cameras(self.scene_path, os.path.join(self.folder_path, "zero12345_narrow_pose.json"))
                
                i_all = np.arange(len(rgb_files))
                i_test = []
                for i, rgb_file in enumerate(rgb_files):
                    if len(os.path.basename(rgb_file).split("_")) == 2 or "_gt" in os.path.basename(rgb_file):
                        i_test.append(i)

                # i_test = np.array(i_test)
                i_train = np.array([idx for idx in i_all if not idx in i_test])
                if mode == "train":
                    del i_test[::self.testskip]
                    i_render = i_test
                else:
                    i_render = i_test[::self.testskip]
                i_render = np.array(i_render)
                
                # i_all = np.arange(len(rgb_files))
                # i_test = []
                # for i, rgb_file in enumerate(rgb_files):
                #     if len(os.path.basename(rgb_file).split("_")) == 2:
                #         # print(rgb_file)
                #         i_test.append(i)
                # i_test = np.array(i_test)
                # i_train = np.array([idx for idx in i_all if not idx in i_test])
                # if mode == "train":
                #     i_render = i_test
                # else:
                #     i_render = i_test
                
                self.train_intrinsics.append(intrinsics[i_train])
                self.train_poses.append(poses[i_train])
                self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
                num_render = len(i_render)

                self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
                self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
                self.render_poses.extend([c2w_mat for c2w_mat in poses[i_render]])
                self.render_train_set_ids.extend([cntr] * num_render)
                cntr += 1
                # print(len(self.train_rgb_files[0]), len(self.render_rgb_files))
                # exit()

        print("loaded {} for {}".format(len(self.render_rgb_files), mode))

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files, train_intrinsics, train_poses = self.train_rgb_files[train_set_id], self.train_intrinsics[train_set_id], self.train_poses[train_set_id]
        train_poses = np.array(train_poses)
        # print(train_poses.shape)

        if self.mode == "train":
            if "_gt" in rgb_file:
                # print("1: ", rgb_file)
                id_render = train_rgb_files.index(rgb_file.replace("_gt", ""))
            else:
                # print("2: ", rgb_file)
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
            # subsample_factor = np.random.choice([0.6, 0.8, 1, 1.2], p=[0.2, 0.2, 0.4, 0.2])
        else:
            id_render = -1
            subsample_factor = 1

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]
        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), render_intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)
        # print(self.num_source_views, subsample_factor)
        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            int(self.num_source_views * subsample_factor),
            tar_id=id_render,
            angular_dist_method="vector",
        )
        # print(len(nearest_pose_ids))
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)
        assert id_render not in nearest_pose_ids
        
        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            # src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        near_depth = 0.5
        far_depth = 1.9

        depth_range = torch.tensor([near_depth, far_depth])

        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
