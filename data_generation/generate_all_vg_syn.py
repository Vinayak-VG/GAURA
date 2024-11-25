import random
import numpy as np
import os
import imageio
import glob
import cv2
import imageio
import numpy as np
import torch
from ibrnet.data_loaders.llff_data_utils import load_llff_data

import os
from PIL import Image
import pandas as pd
from os import listdir
import natsort
import sklearn.preprocessing

from blurgenerator import motion_blur, lens_blur_with_depth_map
from create_haze.create_hazy import gen_haze

def get_image(path):
    img = imageio.imread(path).astype(float)
    return img/255.0

def save_image(img,name):
    img = (img*255.0).astype(np.uint8)
    imageio.imwrite(name,img)
    return



def get_image(path):
    img = imageio.imread(path).astype(float)
    return img/255.0

def save_image(img,name):
    img = (img*255.0).astype(np.uint8)
    imageio.imwrite(name,img)
    return

##### DARK ######

erms = np.random.normal(2.0,0.01)#np.random.uniform(low=1.8, high=2.0, size=None)#np.random.uniform(low=1.0, high=2.0, size=None)
gain = np.random.normal(2.0,0.01)
QE = np.random.uniform(low=0.55, high=0.60, size=None)
blue_gain = np.random.uniform(low=1.0, high=1.05, size=None) #tf.random.uniform((), 1,1.3)
red_gain = np.random.uniform(low=1.0, high=1.3, size=None) #tf.random.uniform((), 1,1.3)

def get_image(path):
    img = imageio.imread(path).astype(float)
    return img/255.0

def save_image(img,name):
    img = (img*255.0).astype(np.uint8)
    imageio.imwrite(name,img)
    return

def random_gains():
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    # rgb_gain = 1.0/np.random.normal(loc=0.0, scale=0.1, size=None) # 1.0 / tf.random.normal((), mean=0.8, stddev=0.1) # random_normal

    # Red and blue gains represent white balance.
    blue_gain = np.random.uniform(low=1.0, high=1.05, size=None) #tf.random.uniform((), 1,1.3)
    red_gain = np.random.uniform(low=1.0, high=1.3, size=None) #tf.random.uniform((), 1,1.3)
    blue_gain = red_gain*blue_gain
    return [red_gain, blue_gain]

def inverse_tonemap(image):
    """Approximately inverts a global tone mapping curve."""
    image = np.clip(image, 0.0, 1.0)
    return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * image) / 3.0)#0.5 - tf.sin(tf.asin(1.0 - 2.0 * image) / 3.0)

def gamma_expansion(image):
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return np.maximum(image, 1e-8) ** 2.2 #tf.maximum(image, 1e-8) ** 2.2

def invert_gains(image, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    gains = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain])# / rgb_gain #tf.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
    gains = gains[np.newaxis, np.newaxis, :] #gains[tf.newaxis, tf.newaxis, :]
    return image * gains

def apply_gains(bayer_image, red_gains, blue_gains):
    """Applies white balance gains to a SINGLE Bayer image."""
    # bayer_images.shape.assert_is_compatible_with((None, None, None, 3))
    green_gains = np.ones_like(red_gains)
    gains = np.stack([red_gains, green_gains, blue_gains])
    gains = gains[np.newaxis, np.newaxis, :]
    return bayer_image * gains

def gamma_compression(image, gamma=2.2):
    """Converts from linear to gamma space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return np.maximum(image, 1e-8) ** (1.0 / gamma)

def tone_map(image):
    return 3*image**2 - 2*image**3

def unprocess_single_image(image, red_gain, blue_gain):
    """Unprocesses an image from sRGB to realistic raw data."""
    # image.shape.assert_is_compatible_with([None, None, 3])
    
    image = inverse_tonemap(image)
    image = gamma_expansion(image)
    image = invert_gains(image, red_gain, blue_gain)
    image =  np.clip(image, 0.0, 1.0)
    return image

def process_single_image(image, red_gain, blue_gain):
    """Processes a SINGLE Bayer RGGB images into sRGB images."""
    image = apply_gains(image, red_gain, blue_gain)
    image = np.clip(image, 0.0, 1.0)
    image = gamma_compression(image)
    image = tone_map(image)
    return image

def add_noise_dark(image,erms,gain,QE,scale,noise_prob):
    image=image/scale
    gain = gain*255.0
    if noise_prob == 1:
        sig = np.sqrt(((erms/gain)**2) + (QE/gain * image) + 1e-10)
    else:
        sig = 0
    return np.random.normal(image,sig)

def create_dark(image_path, save_path, scale, noise_prob):
    img = get_image(image_path)
    image_fin = unprocess_single_image(img, red_gain, blue_gain)
    raw_scale_noisy = add_noise_dark(image_fin,erms,gain,QE,scale,noise_prob)
    imgs_raw_scaled_noise_srgb = process_single_image(raw_scale_noisy, red_gain, blue_gain)
    save_image(imgs_raw_scaled_noise_srgb, save_path)

##### MOTION BLUR ######

def motion_gen_blur(img_path, save_path, size, angle):
    img = np.array(Image.open(img_path))
    result = motion_blur(img, size=size, angle=angle)
    result = Image.fromarray(result)
    result.save(save_path)

    
##### NOISE #######

def get_noise_params_train(gain=4):
    
    noise_data = np.load("/home/vinayak/restoration_nerf_finalsprint/data/synthetic_5d_j2_16_noiselevels6_wide_438x202x320x8.npz")
    sig_read_list = np.unique(noise_data['sig_read'])[2:]
    sig_shot_list = np.unique(noise_data['sig_shot'])[2:]

    log_sig_read = np.log10(sig_read_list)
    log_sig_shot = np.log10(sig_shot_list)

    d_read = np.diff(log_sig_read)[0]
    d_shot = np.diff(log_sig_shot)[0]

    gain_log = np.log2(gain)

    sigma_read = 10 ** (log_sig_read[0] + d_read * gain_log)
    sigma_shot = 10 ** (log_sig_shot[0] + d_shot * gain_log)
    
    return sigma_read, sigma_shot

def get_std(rgb, sig_read, sig_shot):
    return (sig_read ** 2 + sig_shot ** 2 * rgb) ** 0.5

def add_noise(rgb, gain=4):
    sig_read, sig_shot = get_noise_params_train(gain=gain)
    std = get_std(rgb, sig_read, sig_shot)
    noise = std * np.random.randn(*(rgb.shape))
    noise_rgb = rgb + noise
    return noise_rgb

def re_linearize(rgb, wl=1.):
    return wl * (rgb ** 1.3)

def burst_noise(image_path, save_path, gain=4.0, white_level=1):
    img = np.array(Image.open(image_path)) / 255

    img = re_linearize(img[..., :3], white_level)

    img = add_noise(img, gain=gain)
    img = Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8))

    img.save(save_path)
    
###### RAIN #######

def add_gaussian_noise(image, mean, variance):
    h, w, c = image.shape
    gaussian_noise = np.random.normal(mean, np.sqrt(variance), (h, w, c))
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def render_rain(img, theta, density):
    image_rain = img.copy()
    h, w, _ = img.shape

    # Parameter seed generation
    s = 1.01 + 5 * 0.2
    intensity = 1.0
    m = density * (0.2 + 5 * 0.05)  # Mean of Gaussian, controls density of rain
    v = intensity + 5 * 0.3
    l = 20
    # Generate proper noise seed
    dense_chnl = np.zeros((h, w, 1), dtype=np.float32)
    dense_chnl_noise = add_gaussian_noise(dense_chnl, m, v)
    dense_chnl_noise = cv2.resize(dense_chnl_noise, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    posv = np.random.randint(0, dense_chnl_noise.shape[0] - h + 1)
    posh = np.random.randint(0, dense_chnl_noise.shape[1] - w + 1)
    dense_chnl_noise = dense_chnl_noise[posv:posv+h, posh:posh+w]

    kernel_size = l
    angle = theta
    motion_blur_filter = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    motion_blur_filter[int((kernel_size - 1) / 2), :] = np.ones(kernel_size, dtype=np.float32)
    rotation_matrix = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    motion_blur_filter = cv2.warpAffine(motion_blur_filter, rotation_matrix, (kernel_size, kernel_size), flags=cv2.INTER_LINEAR)

    # Apply motion blur
    dense_chnl_motion = cv2.filter2D(dense_chnl_noise, -1, motion_blur_filter)

    # Generate streak with motion blur
    dense_chnl_motion[dense_chnl_motion < 0] = 0
    dense_streak = dense_chnl_motion[:, :, np.newaxis] * np.ones(3, dtype=dense_chnl_motion.dtype)

    # Render Rain streak
    tr = np.random.uniform(0.5, 0.2 + 0.04 * l + 0.05)*0.1
    image_rain += tr * dense_streak
    image_rain = np.clip(image_rain, 0, 1)
    actual_streak = image_rain - img
    return image_rain, actual_streak

def create_rain(rgb_file, save_path, theta, density):
    img = np.array(Image.open(rgb_file)).astype(np.float32) / 255.0
    seed = min(1, abs(np.random.normal(0.5, 0.5)))
    im = cv2.GaussianBlur(img, (0, 0), seed)
    rain, _ = render_rain(im, theta, density)
    rain = Image.fromarray((rain*255).astype(np.uint8))
    rain.save(save_path)

###### SNOW #######
import imgaug.augmenters as iaa
def snow_sim(img_path, save_path, density=0.05, density_uniform=0.8, flake_size=0.6, flake_size_uniformity=0.5, angle=20, speed=0.02):
    
    image = np.array(Image.open(img_path))
    aug = iaa.Snowflakes(density=(density), density_uniformity=(density_uniform),
                flake_size=(flake_size), flake_size_uniformity=(flake_size_uniformity),
                angle=(angle), speed=(speed), seed=45)
    # aug = iaa.Snowflakes(density=(0.1), density_uniformity=(0.3),
    #             flake_size=(0.8), flake_size_uniformity=(0.8),
    #             angle=(-30), speed=(0.001),seed=45)
    image = aug.augment_image(image)
    # Please enter the directory where the synthetic snow images should be saved
    image = Image.fromarray((image).astype(np.uint8))
    image.save(save_path)


def nerfllff(base_dir="/home/vinayak/restoration_nerf_finalsprint/nerf_llff_data"):
    all_rgb_files = []
    scenes = os.listdir(base_dir)
    for i, scene in enumerate(scenes):
        scene_path = os.path.join(base_dir, scene)
        _, _, _, _, _, rgb_files = load_llff_data(
            scene_path, load_imgs=False, factor=4
        )
        all_rgb_files.append(rgb_files)

    return all_rgb_files
    

from midasextractor import MiDaSExtractor
def get_depth(image_path, net=None):
    img = Image.open(image_path).convert("RGB")
    depth = net(img)
    return depth

@torch.no_grad()
def main():
    ckpt_path = "weights/ce_zerodce.pth"

    depth_net = MiDaSExtractor(device="cuda")

    def create_corruptions(rgb_files):
        for scene in rgb_files:
            print("\n")
            # white_level = 1
            # theta = np.random.randint(60, 120)
            # blur_intensity = random.random()
            for i, rgb_file in enumerate(scene):
                
                img_name = os.path.basename(rgb_file)
                dir_path = os.path.dirname(rgb_file)
                
                cor_dir = dir_path + "_dark"
                os.makedirs(cor_dir, exist_ok=True)
                save_path = os.path.join(cor_dir, img_name)
                create_dark(rgb_file, save_path, 30, 1)

                # cor_dir = dir_path + "_motion"
                # os.makedirs(cor_dir, exist_ok=True)
                # save_path = os.path.join(cor_dir, img_name)
                # size = np.random.randint(5, 26)
                # angle_list = list(range(0, 10)) + list(range(80, 100)) + list(range(170, 180))
                # angle = random.choice(angle_list)
                # motion_gen_blur(rgb_file, save_path, size, angle)             

                # depth = get_depth(rgb_file, depth_net)
                # depth = 1 - depth
                # cor_dir = dir_path + "_depth"
                # os.makedirs(cor_dir, exist_ok=True)
                # save_path = os.path.join(cor_dir, img_name)
                # cv2.imwrite(save_path, depth*255.)

                # cor_dir = dir_path + "_haze_test"
                # os.makedirs(cor_dir, exist_ok=True)
                # save_path = os.path.join(cor_dir, img_name)
                # depth_path = rgb_file.replace("images_2", "images_2_depth")
                # gen_haze(rgb_file, save_path, depth_path, 2.5, 128)

                # cor_dir = dir_path + "_noise"
                # os.makedirs(cor_dir, exist_ok=True)
                # save_path = os.path.join(cor_dir, img_name)
                # burst_noise(rgb_file, save_path, gain=24, white_level=1)

                # cor_dir = dir_path + "_rain"
                # os.makedirs(cor_dir, exist_ok=True)
                # save_path = os.path.join(cor_dir, img_name)
                # create_rain(rgb_file, save_path, -80, -4)

                # cor_dir = dir_path + "_snow"
                # os.makedirs(cor_dir, exist_ok=True)
                # save_path = os.path.join(cor_dir, img_name)
                # snow_sim(rgb_file, save_path, density=0.07, density_uniform=0.9, flake_size=0.7, flake_size_uniformity=0.4, angle=-20, speed=0.01)

                print(i, end="\r")

    rgb_files = nerfllff()
    create_corruptions(rgb_files)
    print("completed nerf llff")

if __name__ == "__main__":
    main()