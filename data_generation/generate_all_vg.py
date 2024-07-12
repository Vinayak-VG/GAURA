import random
import numpy as np
import os
import imageio
import glob
import cv2
import imageio
import numpy as np
import torch
from llff_data_utils import load_llff_data
# from BLUR.imagenet_c_new import corrupt
import os
from PIL import Image
import pandas as pd
from os import listdir
import natsort
import sklearn.preprocessing

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

def unprocess(imgs):
    metadata = random_gains()
    return [unprocess_single_image(x, metadata[0],metadata[1]) for x in imgs], metadata

def process(imgs,metadata):
    return [process_single_image(x,metadata[0],metadata[1]) for x in imgs]

def get_noise_params_train(gain=4):
    
    noise_data = np.load("./synthetic_5d_j2_16_noiselevels6_wide_438x202x320x8.npz")
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
    img = cv2.imread(image_path) / 255

    img = re_linearize(img[..., :3], white_level)

    img = add_noise(img, gain=gain)
    cv2.imwrite(save_path, img * 255)

def create_image(J, betaD, Binf, betaB, depth, z):
    I = np.zeros(J.shape)
    
    z = np.expand_dims(z, axis = -1)
    z = np.repeat(z, 3, axis = -1)
    I[:,:,0] = (J[:,:,0] * np.exp((-1 * betaD[2] * depth * z[:,:,0]))) + (Binf[:,:,0] * (1 - np.exp((-1 * betaB[2] * z[:,:,0]))))
    I[:,:,1] = (J[:,:,1] * np.exp((-1 * betaD[1] * depth * z[:,:,1]))) + (Binf[:,:,1] * (1 - np.exp((-1 * betaB[1] * z[:,:,1]))))
    I[:,:,2] = (J[:,:,2] * np.exp((-1 * betaD[0] * depth * z[:,:,2]))) + (Binf[:,:,2] * (1 - np.exp((-1 * betaB[0] * z[:,:,2]))))
    return I

# Define the input and output direct

# Function to process an image

erms = np.random.normal(2.0,0.01)#np.random.uniform(low=1.8, high=2.0, size=None)#np.random.uniform(low=1.0, high=2.0, size=None)
gain = np.random.normal(2.0,0.01)
QE = np.random.uniform(low=0.55, high=0.60, size=None)
blue_gain = np.random.uniform(low=1.0, high=1.05, size=None) #tf.random.uniform((), 1,1.3)
red_gain = np.random.uniform(low=1.0, high=1.3, size=None) #tf.random.uniform((), 1,1.3)


def add_noise_dark(image,erms,gain,QE,scale):
    image=image/scale
    gain = gain*255.0
    sig = np.sqrt(((erms/gain)**2) + (QE/gain * image) + 1e-10)
    return np.random.normal(image,sig)


def process_image(image_path,save_path,scale):
    img = get_image(image_path)
    image_fin = unprocess_single_image(img, red_gain, blue_gain)
    # scale = 20
    raw_scale_noisy = add_noise_dark(image_fin,erms,gain,QE,scale)

    imgs_raw_scaled_noise_srgb = process_single_image(raw_scale_noisy, red_gain, blue_gain)

    save_image(imgs_raw_scaled_noise_srgb,save_path)

from midasextractor import MiDaSExtractor
def get_depth(image_path, net=None):
    img = Image.open(image_path).convert("RGB")
    depth = net(img)
    return depth
 
def iconic(base_dir="/home/vinayak/restoration_nerf_finalsprint/data/real_iconic_noface/"):
    all_rgb_files = []
    scenes = os.listdir(base_dir)
    for i, scene in enumerate(scenes):
        scene_path = os.path.join(base_dir, scene)
        _, _, _, _, _, rgb_files = load_llff_data(
            scene_path, load_imgs=False, factor=4
        )
        all_rgb_files.append(rgb_files)
    return all_rgb_files

def nerfllff(base_dir="/home/vinayak/restoration_nerf_finalsprint/real_data/haze/final_revide"):
    all_rgb_files = []
    scenes = os.listdir(base_dir)
    for i, scene in enumerate(scenes):
        if "W001" in scene:
            scene_path = os.path.join(base_dir, scene)
            _, _, _, _, _, rgb_files = load_llff_data(
                scene_path, load_imgs=False, factor=2
            )
            all_rgb_files.append(rgb_files)
    return all_rgb_files

def ibrnet_collection(folder_path1="/home/vinayak/restoration_nerf_finalsprint/data/ibrnet_collected_1/", folder_path2="/home/vinayak/restoration_nerf_finalsprint/data/ibrnet_collected_2/"):
    all_scenes = glob.glob(folder_path1 + "*") + glob.glob(folder_path2 + "*")
    all_rgb_files = []
    for i, scene in enumerate(all_scenes):
        if "ibrnet_collected_2" in scene:
            factor = 8
        else:
            factor = 2
        _, _, _, _, _, rgb_files = load_llff_data(
            scene, load_imgs=False, factor=factor
        )
        all_rgb_files.append(rgb_files)
    return all_rgb_files

@torch.no_grad()
def main():

    depth_net = MiDaSExtractor(device="cuda")

    def create_corruptions(rgb_files):
        for scene in rgb_files:
            print("\n")

            for i, rgb_file in enumerate(scene):
                
                img_name = os.path.basename(rgb_file)
                dir_path = os.path.dirname(rgb_file)

                depth = get_depth(rgb_file, depth_net)
                depth = 1 - depth
                cor_dir = dir_path + "_depth"
                os.makedirs(cor_dir, exist_ok=True)
                save_path = os.path.join(cor_dir, img_name)
                cv2.imwrite(save_path, depth*255.)
                
                cor_dir = dir_path + "_haze"
                os.makedirs(cor_dir, exist_ok=True)
                save_path = os.path.join(cor_dir, img_name)
                depth_path = rgb_file.replace("images_4", "images_4_depth")
                gen_haze(rgb_file, save_path, depth_path, 5, 225)

                print(i, end="\r")
                
    # rgb_files = ibrnet_collection()
    # print(rgb_files)
    # create_corruptions(rgb_files)
    # print("completed ibrnet collected")
    # rgb_files = iconic()
    # create_corruptions(rgb_files)
    # print("completed real iconic")
    rgb_files = nerfllff()
    create_corruptions(rgb_files)
    print("completed nerf llff")
    
if __name__ == "__main__":
    main()