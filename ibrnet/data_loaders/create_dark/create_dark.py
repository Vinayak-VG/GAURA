import numpy as np
import imageio

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

def process_image(image_path, scale, noise_prob):
    img = get_image(image_path)
    image_fin = unprocess_single_image(img, red_gain, blue_gain)
    raw_scale_noisy = add_noise_dark(image_fin,erms,gain,QE,scale,noise_prob)

    imgs_raw_scaled_noise_srgb = process_single_image(raw_scale_noisy, red_gain, blue_gain)
    return imgs_raw_scaled_noise_srgb

if __name__ == "__main__":
    from PIL import Image
    result = process_image("/data/cilab4090/input.JPG", 30, 1) * 255
    result = Image.fromarray(result.astype(np.uint8))
    result.save("/data/cilab4090/result_dark.png")