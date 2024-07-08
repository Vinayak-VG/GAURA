from MiDaS.midas.model_loader import default_models, load_model
import torch.nn.functional as F
from torchvision import transforms
import torch
import numpy as np


class MiDaSExtractor:
    def __init__(self, device=None):
        self.model, self.transform, net_w, net_h = load_model(device, "data_generation/MiDaS/weights/dpt_swin2_large_384.pt", "dpt_swin2_large_384", False, None, False)
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, img, bits=1):
        img = self.to_tensor(img)[None, :, :, :]
        orig_size = img.shape[2:]
        width, height = self.transform.transforms[0].get_size(
            img.shape[2], img.shape[1]
        )
        img = F.interpolate(img, size=(height, width), mode="bicubic", align_corners=False)
        img = self.norm(img)
        depth = self.model.forward(img.to(self.device))
        depth = depth[:, None, :, :]
        depth = F.interpolate(depth, size=orig_size, mode="bicubic", align_corners=False)
        depth = depth.squeeze().detach().cpu().numpy()

        depth_min = depth.min()
        depth_max = depth.max()
        # max_val = (2**(8*bits))-1
        max_val = 1
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)
        # out = out * 0.2 + 0.4
        # out = depth
        # print(depth_min, depth_max)
        return out