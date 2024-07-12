import torch
import random
import os
import numpy as np
from ibrnet.data_loaders.create_haze.create_hazy import gen_haze
from ibrnet.data_loaders.create_dark.create_dark import process_image
from ibrnet.data_loaders.create_noise.create_noise import burst_noise
from ibrnet.data_loaders.create_rain.create_rain import create_rain

degradation = "haze"
nearest_pose_ids = None
train_rgb_files = None

if degradation == "haze":
    beta = random.randrange(2, 9) / 2.
    A_light = random.randint(150, 255)
    for id in nearest_pose_ids:
        src_rgb_file = train_rgb_files[id]
        src_rgb_file_depth = os.path.join(os.path.dirname(src_rgb_file) + "_depth", os.path.basename(src_rgb_file))
        # depth_map = imageio.imread(src_rgb_file_depth).astype(np.float32) / 255.0
        src_rgb_file = os.path.join(os.path.dirname(src_rgb_file) + "", os.path.basename(src_rgb_file))
        src_rgb = (gen_haze(src_rgb_file, src_rgb_file_depth, beta, A_light) / 255.0).astype(np.float32)
elif degradation == "dark":
    scale = random.randint(6, 26)
    for id in nearest_pose_ids:
        src_rgb_file = train_rgb_files[id]
        src_rgb_file = os.path.join(os.path.dirname(src_rgb_file) + "", os.path.basename(src_rgb_file))
        src_rgb = process_image(src_rgb_file, scale)
elif degradation == "noise":
    gain = random.randint(10, 30)
    for id in nearest_pose_ids:
        src_rgb_file = train_rgb_files[id]
        src_rgb_file = os.path.join(os.path.dirname(src_rgb_file) + "", os.path.basename(src_rgb_file))
        src_rgb = burst_noise(src_rgb_file, gain=gain, white_level=1)
elif degradation == "rain":
    gain = random.randint(10, 30)
    theta = np.random.randint(60, 120)
    density = random.randint(-3, -7)
    for id in nearest_pose_ids:
        src_rgb_file = train_rgb_files[id]
        src_rgb_file = os.path.join(os.path.dirname(src_rgb_file) + "", os.path.basename(src_rgb_file))
        src_rgb = create_rain(src_rgb_file, theta, a=0, density=density, intensity=1.0)
    
    
    