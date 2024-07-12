import cv2
import numpy as np
from PIL import Image

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
    # v = intensity + np.random.rand() * 0.3  # Variance of Gaussian, controls intensity of rain streak
    v = intensity + 5 * 0.3
    # l = np. om.randint(20, 60)  # Length of motion blur, controls size of rain streak
    l = 20
    # Generate proper noise seed
    dense_chnl = np.zeros((h, w, 1), dtype=np.float32)
    dense_chnl_noise = add_gaussian_noise(dense_chnl, m, v)
    # dense_chnl_noise = cv2.resize(dense_chnl_noise, (w, h), interpolation=cv2.INTER_CUBIC)
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
    # dense_chnl_motion = cv2.filter2D(dense_chnl_noise, -1, filter)
    dense_chnl_motion = cv2.filter2D(dense_chnl_noise, -1, motion_blur_filter)

    # Generate streak with motion blur
    dense_chnl_motion[dense_chnl_motion < 0] = 0
    # dense_streak = np.repeat(dense_chnl_motion[:, :, np.newaxis], 3, axis=2)
    dense_streak = dense_chnl_motion[:, :, np.newaxis] * np.ones(3, dtype=dense_chnl_motion.dtype)

    # Render Rain streak
    # tr = np.random.uniform(0.04 * l + 0.2, 0.09 * l + 0.2)
    tr = np.random.uniform(0.5, 0.2 + 0.04 * l + 0.05)*0.1
    image_rain += tr * dense_streak
    # image_rain = np.add(image_rain, tr * dense_streak, out=image_rain, casting="unsafe")
    image_rain = np.clip(image_rain, 0, 1)
    actual_streak = image_rain - img
    return image_rain, actual_streak

def create_rain(rgb_file, theta, density):
    img = np.array(Image.open(rgb_file)).astype(np.float32) / 255.0
    seed = min(1, abs(np.random.normal(0.5, 0.5)))
    im = cv2.GaussianBlur(img, (0, 0), seed)
    rain, _ = render_rain(im, theta, density)
    
    return rain


if __name__ == "__main__":
    result = create_rain("/data/cilab4090/input.JPG", 120, -6) * 255
    result = Image.fromarray(result.astype(np.uint8))
    result.save("/data/cilab4090/result_rain.png")