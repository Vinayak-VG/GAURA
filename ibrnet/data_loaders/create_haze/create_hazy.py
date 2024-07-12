import random
import numpy as np
import os
import imageio
import glob
import cv2
import imageio
import numpy as np
import torch
import os
from PIL import Image
import pandas as pd
from os import listdir
import natsort
import sklearn.preprocessing
import ibrnet.data_loaders.create_haze.tool_kit as tk
from ibrnet.data_loaders.create_haze.parameter import const

def gen_haze(img_path, depth_path, Beta, A):
    img = np.array(Image.open(img_path))
    depth_img = np.array(Image.open(depth_path)).astype(np.float64)
    # depth_img_3c = np.zeros_like(img)
    # depth_img_3c[:,:,0] = depth_img
    # depth_img_3c[:,:,1] = depth_img
    # depth_img_3c[:,:,2] = depth_img
    depth_img_3c = np.repeat(np.expand_dims(depth_img, axis = -1), 3, axis = -1)
    norm_depth_img = depth_img_3c/255.
    norm_depth_img = cv2.normalize(norm_depth_img,None,0.2,1.0,cv2.NORM_MINMAX,5)
    # norm_depth_img = 1 - norm_depth_img
    trans = np.exp(-norm_depth_img*Beta)

    hazy = img*trans + A*(1-trans)
    hazy = np.array(hazy, dtype=np.uint8)
    
    return hazy

# def haze_image(image_path, depth, save_path, parameter):
#     img = image_path
#     Ip = cv2.imread(img)
#     # depth[depth==0] = 1
#     # depth *= 3

#     I = np.empty_like(Ip)
#     result = np.empty_like(Ip)

#     elevation, distance, angle = tk.elevation_and_distance_estimation(img, depth,
#                                                             const.CAMERA_VERTICAL_FOV,
#                                                             const.HORIZONTAL_ANGLE,
#                                                             const.CAMERA_ALTITUDE)

#     ecm = parameter / const.VISIBILITY_RANGE_MOLECULE
#     eca = parameter / const.VISIBILITY_RANGE_AEROSOL

#     if const.FT != 0:
#         perlin = tk.noise(Ip, depth)
#         ECA = eca
#         # ECA = eca * np.exp(-elevation/(const.FT+0.00001))
#         c = (1-elevation/(const.FT+0.00001))
#         c[c<0] = 0

#         if const.FT > const.HT:
#             ECM = (ecm * c + (1-c)*eca) * (perlin/255)
#         else:
#             ECM = (eca * c + (1-c)*ecm) * (perlin/255)

#     else:
#         ECA = eca
#         # ECA = eca * np.exp(-elevation/(const.FT+0.00001))
#         ECM = ecm


#     distance_through_fog = np.zeros_like(distance)
#     distance_through_haze = np.zeros_like(distance)
#     distance_through_haze_free = np.zeros_like(distance)


#     if const.FT == 0:  # only haze: const.FT should be set to 0
#         idx1 = elevation > const.HT
#         idx2 = elevation <= const.HT

#         if const.CAMERA_ALTITUDE <= const.HT:
#             distance_through_haze[idx2] = distance[idx2]
#             distance_through_haze_free[idx1] = (elevation[idx1] - const.HT) * distance[idx1] \
#                                             / (elevation[idx1] - const.CAMERA_ALTITUDE)

#             distance_through_haze[idx1] = distance[idx1] - distance_through_haze_free[idx1]

#         elif const.CAMERA_ALTITUDE > const.HT:
#             distance_through_haze_free[idx1] = distance[idx1]
#             distance_through_haze[idx2] = (const.HT - elevation[idx2]) * distance[idx2] \
#                                         / (const.CAMERA_ALTITUDE - elevation[idx2])
#             distance_through_haze_free[idx2] = distance[idx2] - distance_through_fog[idx2]

#         I[:, :, 0] = Ip[:, :, 0] * np.exp(-ECA*distance_through_haze-ECM*distance_through_haze_free)
#         I[:, :, 1] = Ip[:, :, 1] * np.exp(-ECA*distance_through_haze-ECM*distance_through_haze_free)
#         I[:, :, 2] = Ip[:, :, 2] * np.exp(-ECA*distance_through_haze-ECM*distance_through_haze_free)
#         O = 1-np.exp(-ECA*distance_through_haze-ecm*distance_through_haze_free)


#     elif const.FT < const.HT and const.FT != 0:
#         idx1 = (np.logical_and(const.HT > elevation, elevation > const.FT))
#         idx2 = elevation <= const.FT
#         idx3 = elevation >= const.HT
#         if const.CAMERA_ALTITUDE <= const.FT:
#             distance_through_fog[idx2] = distance[idx2]
#             distance_through_haze[idx1] = (elevation[idx1] - const.FT) * distance[idx1] \
#                                             / (elevation[idx1] - const.CAMERA_ALTITUDE)

#             distance_through_fog[idx1] = distance[idx1] - distance_through_haze[idx1]
#             distance_through_fog[idx3] = (const.FT - const.CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
#             distance_through_haze[idx3] = (const.HT - const.FT) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
#             distance_through_haze_free[idx3] = distance[idx3] - distance_through_haze[idx3] - distance_through_fog[idx3]

#         elif const.CAMERA_ALTITUDE > const.HT:
#             distance_through_haze_free[idx3] = distance[idx3]
#             distance_through_haze[idx1] = (const.FT - elevation[idx1]) * distance_through_haze_free[idx1] \
#                                         / (const.CAMERA_ALTITUDE - const.HT)
#             distance_through_haze_free[idx1] = distance[idx1] - distance_through_haze[idx1]


#             distance_through_fog[idx2] = (const.FT - elevation[idx2]) * distance[idx2] / (const.CAMERA_ALTITUDE - elevation[idx2])
#             distance_through_haze[idx2] = (const.HT - const.FT) * distance / (const.CAMERA_ALTITUDE - elevation[idx2])
#             distance_through_haze_free[idx2] = distance[idx2] - distance_through_haze[idx2] - distance_through_fog[idx2]

#         elif const.FT < const.CAMERA_ALTITUDE <= const.HT:
#             distance_through_haze[idx1] = distance[idx1]
#             distance_through_fog[idx2] = (const.FT - elevation[idx2]) * distance[idx2] / (const.CAMERA_ALTITUDE - elevation[idx2])
#             distance_through_haze[idx2] = distance[idx2] - distance_through_fog[idx2]
#             distance_through_haze_free[idx3] = (elevation[idx3] - const.HT) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
#             distance_through_haze[idx3] = (const.HT - const.CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)

#         I[:, :, 0] = Ip[:, :, 0] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
#         I[:, :, 1] = Ip[:, :, 1] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
#         I[:, :, 2] = Ip[:, :, 2] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
#         O = 1-np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)


#     elif const.FT > const.HT:
#         if const.CAMERA_ALTITUDE <= const.HT:
#             idx1 = (np.logical_and(const.FT > elevation, elevation > const.HT))
#             idx2 = elevation <= const.HT
#             idx3 = elevation >= const.FT

#             distance_through_haze[idx2] = distance[idx2]
#             distance_through_fog[idx1] = (elevation[idx1] - const.HT) * distance[idx1] \
#                                             / (elevation[idx1] - const.CAMERA_ALTITUDE)
#             distance_through_haze[idx1] = distance[idx1] - distance_through_fog[idx1]
#             distance_through_haze[idx3] = (const.HT - const.CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
#             distance_through_fog[idx3] = (const.FT - const.HT) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
#             distance_through_haze_free[idx3] = distance[idx3] - distance_through_haze[idx3] - distance_through_fog[idx3]

#         elif const.CAMERA_ALTITUDE > const.FT:
#             idx1 = (np.logical_and(const.HT > elevation, elevation > const.FT))
#             idx2 = elevation <= const.FT
#             idx3 = elevation >= const.HT

#             distance_through_haze_free[idx3] = distance[idx3]
#             distance_through_haze[idx1] = (const.FT - elevation[idx1]) * distance_through_haze_free[idx1] \
#                                         / (const.CAMERA_ALTITUDE - const.HT)
#             distance_through_haze_free[idx1] = distance[idx1] - distance_through_haze[idx1]
#             distance_through_fog[idx2] = (const.FT - elevation[idx2]) * distance[idx2] / (const.CAMERA_ALTITUDE - elevation[idx2])
#             distance_through_haze[idx2] = (const.HT - const.FT) * distance / (const.CAMERA_ALTITUDE - elevation[idx2])
#             distance_through_haze_free[idx2] = distance[idx2] - distance_through_haze[idx2] - distance_through_fog[idx2]

#         elif const.HT < const.CAMERA_ALTITUDE <= const.FT:
#             idx1 = (np.logical_and(const.HT > elevation, elevation > const.FT))
#             idx2 = elevation <= const.FT
#             idx3 = elevation >= const.HT

#             distance_through_haze[idx1] = distance[idx1]
#             distance_through_fog[idx2] = (const.FT - elevation[idx2]) * distance[idx2] / (const.CAMERA_ALTITUDE - elevation[idx2])
#             distance_through_haze[idx2] = distance[idx2] - distance_through_fog[idx2]
#             distance_through_haze_free[idx3] = (elevation[idx3] - const.HT) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
#             distance_through_haze[idx3] = (const.HT - const.CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)

#         I[:, :, 0] = Ip[:, :, 0] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
#         I[:, :, 1] = Ip[:, :, 1] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
#         I[:, :, 2] = Ip[:, :, 2] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
#         O = 1-np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)

#     Ial = np.empty_like(Ip)  # color of the fog/haze
#     Ial[:, :, 0] = 255
#     Ial[:, :, 1] = 255
#     Ial[:, :, 2] = 255
#     # Ial[:, :, 0] = 240
#     # Ial[:, :, 1] = 240
#     # Ial[:, :, 2] = 240

#     result[:, :, 0] = I[:, :, 0] + O * Ial[:, :, 0]
#     result[:, :, 1] = I[:, :, 1] + O * Ial[:, :, 1]
#     result[:, :, 2] = I[:, :, 2] + O * Ial[:, :, 2]
    
#     return result