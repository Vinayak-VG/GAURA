# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .google_scanned_objects import *
from .realestate import *
from .deepvoxels import *
from .realestate import *
from .llff import *
from .llff_test import *
from .ibrnet_collected import *
from .realestate import *
from .spaces_dataset import *
from .nerf_synthetic import *
from .shiny import *
from .llff_render import *
from .shiny_render import *
from .nerf_synthetic_render import *
from .nmr_dataset import *
from .objaverse_zero123 import *
from .co3d_loader_splits import *
from .co3d_loader import *
from .raven import *
from .objaverse_zero12345 import *
# from .aleth_lom import *
from .deblur_real_cam_motion import *
from .deblur_syn_cam_motion import *
from .seathru_nerf import *
from .llnerf_dataloader import *
from .aleth_nerf import *
from .deblur_real_defocus import *
from .dehazenerf import *
from .llff_noise import *
from .revide import *
from .realrain import *
from .dwarka import *
from .ibrnet_collected_haze import *
from .llff_haze import *
from .llff_test_haze import *
from .ibrnet_collected_blur import *
from .llff_blur import *
from .llff_test_blur import *
from .ibrnet_collected_dyn import *
from .llff_dyn import *
from .llff_test_dyn import *
from .realsnow import *
from .llff_test_multi import *
from .nannerf import *
from .scene_test import *

dataset_dict = {
    "spaces": SpacesFreeDataset,
    "google_scanned": GoogleScannedDataset,
    "realestate": RealEstateDataset,
    "deepvoxels": DeepVoxelsDataset,
    "nerf_synthetic": NerfSyntheticDataset,
    "llff": LLFFDataset,
    "ibrnet_collected": IBRNetCollectedDataset,
    "llff_test": LLFFTestDataset,
    "shiny": ShinyDataset,
    "llff_render": LLFFRenderDataset,
    "shiny_render": ShinyRenderDataset,
    "nerf_synthetic_render": NerfSyntheticRenderDataset,
    "nmr": NMRDataset,
    "objaversezero123": ObjaverseZero123Dataset,
    "co3d": CO3DDataset,
    "co3d_splits": CO3DSplitDataset,
    "raven": RavenDataset,
    "objaversezero12345": ObjaverseZero12345Dataset,
    "deblurrealmotion": DeblurRealMotion,
    "deblursynmotion": DeblurSynMotion,
    "seathru_nerf": Seathru_Water,
    "llnerf": LLNeRFDataloader,
    "aleth_nerf": AlethLOMDataset,
    "deblurrealdefocus": DeblurRealDefocus,
    "dehazenerfreal": DehazeNeRFReal,
    "llff_noise": LLFFNoiseDataset,
    "revide": RevideDataloader,
    "realrain": RealRainDataloader,
    "dwarka": DwarkaDataset,
    "ibrnet_collected_haze": IBRNetCollectedHazeDataset,
    "llff_test_haze": LLFFTestHazeDataset,
    "llff_haze": LLFFHazeDataset,
    "ibrnet_collected_blur": IBRNetCollectedBlurDataset,
    "llff_test_blur": LLFFTestBlurDataset,
    "llff_blur": LLFFBlurDataset,
    "ibrnet_collected_dyn": IBRNetCollectedDatasetDynamic,
    "llff_test_dyn": LLFFTestDatasetDynamic,
    "llff_dyn": LLFFDatasetDynamic,
    "realsnow": RealSnowDataloader,
    "multi_corr": NeRFLLFFMultiDataset, 
    "nannerf": NANNeRFNoiseDataloader,
    "scene_test": SceneTestDataloader,
}

