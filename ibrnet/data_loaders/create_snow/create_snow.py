import glob
import cv2
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np

# Please enter the directory of the input ground truth image sequences

def snow_sim(img_path, density=0.08, density_uniform=0.9, flake_size=0.2, flake_size_uniformity=0.4, angle=20, speed=0.01):
    
    image = np.array(Image.open(img_path))
    # image = cv2.imread(img_path)
    aug = iaa.Snowflakes(density=(density), density_uniformity=(density_uniform),
                flake_size=(flake_size), flake_size_uniformity=(flake_size_uniformity),
                angle=(angle), speed=(speed),seed=45)
    # aug = iaa.Snowflakes(density=(0.1), density_uniformity=(0.3),
    #             flake_size=(0.8), flake_size_uniformity=(0.8),
    #             angle=(-30), speed=(0.001),seed=45)
    image = aug.augment_image(image)
    # Please enter the directory where the synthetic snow images should be saved
    return image/255.

if __name__ == "__main__":
    import random
    # angle = random.randrange(-30, 30, 5)
    # speed = random.choice([0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04])
    # flake_size = random.random() * 0.6 + 0.1
    # density_uniform = random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # flake_size_uniformity = random.choice([0.4,0.5,0.6,0.8])
    # density = random.random() * 0.070 + 0.005
    result = snow_sim("/data/cilab4090/1.JPG", ) * 255
    result = Image.fromarray(result.astype(np.uint8))
    result.save("/data/cilab4090/result_snow.png")
