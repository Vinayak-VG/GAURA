from PIL import Image
import os
import sys

directory = "/home/vinayak/restoration_nerf_finalsprint/real_data/haze/final_revide/E006/images_1_haze"
save_directory = "/home/vinayak/restoration_nerf_finalsprint/real_data/haze/final_revide/E006/images_2_haze"
os.makedirs(save_directory, exist_ok=True)

for file_name in os.listdir(directory):
    print("Processing %s" % file_name)
    image = Image.open(os.path.join(directory, file_name))

    x,y = image.size
    new_dimensions = (x//2, y//2) #dimension set here
    output = image.resize(new_dimensions)

    output_file_name = os.path.join(save_directory, file_name)
    output.save(output_file_name)

# print("All done")