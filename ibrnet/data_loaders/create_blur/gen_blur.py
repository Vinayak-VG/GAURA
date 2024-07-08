import cv2
from blurgenerator import motion_blur, lens_blur_with_depth_map
from PIL import Image
import numpy as np
def motion_gen_blur(img_path, size, angle, blur_type):
    if blur_type == "motion":
        img = np.array(Image.open(img_path))
        result = motion_blur(img, size=size, angle=angle)
    return result

def defocus_gen_blur(img_path, depth_path, blur_amount, depth_type):
    # blur_amount range = 5 to 20 & Change Depth to normal depth and inverse depth 50% of the time
    img = np.array(Image.open(img_path))
    depth_img = np.repeat(np.expand_dims(np.array(Image.open(depth_path)), -1), 3, -1).astype(np.uint8)
    # print(depth_img.dtype)
    # quit()
    if depth_type == -1:
        depth_img = 255 - depth_img
    result = lens_blur_with_depth_map(
    img,
    depth_map=depth_img,
    components=5,
    exposure_gamma=5,
    num_layers=10,
    min_blur=1,
    max_blur=blur_amount
    )
    return result
    
"""MOTION BLUR"""

if __name__ == "__main__":
    result = motion_gen_blur("/data/cilab4090/input.JPG", 30, 90, "motion")
    result = Image.fromarray(result)
    result.save("/data/cilab4090/result.png")
# import cv2
# from blurgenerator import motion_blur
# img = cv2.imread('/data/Nerf_ICCV/image012.png')
# result = motion_blur(img, size=40, angle=150)
# cv2.imwrite('/data/Nerf_ICCV/output_size10_angle_10.png', result)

"""MOTION BLUR WITH DEPTH MAP"""

# import cv2
# from blurgenerator import motion_blur_with_depth_map
# img = cv2.imread('/data/Nerf_ICCV/image012.png')
# depth_img = cv2.imread('/data/Nerf_ICCV/image012_depth.png')
# result = motion_blur_with_depth_map(
#    img,
#    depth_map=depth_img,
#    angle=50,
#    num_layers=10,
#    min_blur=1,
#    max_blur=60
# )
# cv2.imwrite('/data/Nerf_ICCV/output_depth_size60_angle_50.png', result)


"""LENS BLUR"""

# import cv2
# from blurgenerator import lens_blur
# img = cv2.imread('/data/Nerf_ICCV/image012.png')
# result = lens_blur(img, radius=10, components=5, exposure_gamma=2)
# cv2.imwrite('/data/Nerf_ICCV/lensblur_output_10_5_2.png', result)


"""Depth based LENS BLUR"""


# import cv2
# from blurgenerator import lens_blur_with_depth_map
# img = cv2.imread('/data/Nerf_ICCV/image012.png')
# depth_img = cv2.imread('/data/Nerf_ICCV/image012_depth.png')
# result = lens_blur_with_depth_map(
#    img,
#    depth_map=depth_img,
#    components=5,
#    exposure_gamma=5,
#    num_layers=10,
#    min_blur=1,
#    max_blur=10
# )
# cv2.imwrite('/data/Nerf_ICCV/lens_blur_depth_10.png', result)