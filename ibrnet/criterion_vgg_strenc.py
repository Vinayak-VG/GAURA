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

import torch.nn as nn
from utils import img2mse
import torch
import torch.nn.functional as F
import torchvision


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).to("cuda").features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).to("cuda").features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).to("cuda").features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).to("cuda").features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to("cuda"))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to("cuda"))
        self.patch = 18

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        input = input.view(1, 3, self.patch, self.patch)
        target = target.view(1, 3, self.patch, self.patch)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    
class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_loss = VGGPerceptualLoss()
    def forward(self, outputs, ray_batch, reg_out, true_reg, scalars_to_log):

        pred_rgb = outputs["rgb"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]
        loss = img2mse(pred_rgb, gt_rgb, pred_mask)
        # vggloss = self.vgg_loss(pred_rgb, gt_rgb)
        # regloss = torch.mean((reg_out - true_reg) ** 2)
        loss = loss# + 0.01*vggloss# + 0.001*regloss

        return loss, scalars_to_log
