import random
import numpy as np
import os
import imageio
import glob
import cv2
import imageio
import numpy as np

from llff_data_utils import load_llff_data
from imagenet_c_new import corrupt

from os import listdir

def get_image(path):
#     img = (imageio.imread(path)[:,64:,:]).astype(float)
    img = imageio.imread(path).astype(float)
    return img/255.0

def save_image(img,name):
    img = (img*255.0).astype(np.uint8)
    imageio.imwrite(name,img)
    return


def llff(base_dir="./data/real_iconic_noface/"):
    all_rgb_files = []
    scenes = os.listdir(base_dir)
    for i, scene in enumerate(scenes):
        scene_path = os.path.join(base_dir, scene)
        _, _, _, _, _, rgb_files = load_llff_data(
            scene_path, load_imgs=False, factor=4
        )
        all_rgb_files.append(rgb_files)
    return all_rgb_files

def nerfllff(base_dir="/data/nerf_llff_data/"):
    all_rgb_files = []
    scenes = os.listdir(base_dir)
    for i, scene in enumerate(scenes):
        scene_path = os.path.join(base_dir, scene)
        _, _, _, _, _, rgb_files = load_llff_data(
            scene_path, load_imgs=False, factor=4
        )
        all_rgb_files.append(rgb_files)
    return all_rgb_files

from llff_data_utils import load_llff_data
def ibrnet_collection(folder_path1="/data/ibrnet_collected_1/", folder_path2="/data/ibrnet_collected_2/"):
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


import os
from PIL import Image





def create_corruptions(rgb_files):
        for scene in rgb_files:
            # print("\n")
            for i, rgb_file in enumerate(scene):
                try:
                    img_name = os.path.basename(rgb_file)
                    dir_path = os.path.dirname(rgb_file)
                
                    cor_dir = dir_path + "_defocus_blur_5/"
                    # print(cor_dir)
                    # quit()
                    os.makedirs(cor_dir, exist_ok=True)
                    save_path = cor_dir + img_name
                    # print(rgb_file)
                    # quit()
                    x = cv2.imread(rgb_file)
                    final = corrupt(x, severity=5, corruption_name= 'defocus_blur' , corruption_number =3)
                    cv2.imwrite(save_path,final)
                    
                except:
                    print("\n")
                    # print(rgb_file)
                
                # print(i, end="\r")
rgb_files = ibrnet_collection()
create_corruptions(rgb_files)
# print("completed nerf llff")


