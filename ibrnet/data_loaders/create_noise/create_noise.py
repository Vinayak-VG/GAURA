import numpy as np
import cv2
from PIL import Image

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

def burst_noise(image_path, gain=4.0, white_level=1):
    img = np.array(Image.open(image_path)) / 255

    img = re_linearize(img[..., :3], white_level)

    img = add_noise(img, gain=gain)
    return img