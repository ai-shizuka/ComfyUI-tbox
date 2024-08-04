import sys
from pathlib import Path
from .utils import here

sys.path.insert(0, str(Path(here, "src").resolve()))

from .nodes.load_node import LoadImageNode
from .nodes.save_node import SaveImageNode
from .nodes.save_node import SaveImagesNode
from .nodes.image_node import ImageResizeNode
from .nodes.image_node import ImageSizeNode
from .nodes.constrain_node import ConstrainImageNode
from .nodes.midas_node import MIDAS_Depth_Map_Preprocessor
from .nodes.dwpose_node import DWPose_Preprocessor, AnimalPose_Preprocessor
from .nodes.densepose_node import DensePose_Preprocessor



print(f'here>> {here} ')

NODE_CLASS_MAPPINGS = {
    "ImageLoader": LoadImageNode,
    "ImageSaver": SaveImageNode,
    "ImagesSaver": SaveImagesNode,
    "ImageResize": ImageResizeNode,
    "ImageSize": ImageSizeNode,
    "ConstrainImageNode": ConstrainImageNode,
    "DensePosePreprocessor": DensePose_Preprocessor,
    "DWPosePreprocessor": DWPose_Preprocessor,
    "AnimalPosePreprocessor": AnimalPose_Preprocessor,
    "MiDaSDepthPreprocessor": MIDAS_Depth_Map_Preprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageLoader": "Image Load",
    "ImageSaver": "Image Save",
    "ImagesSaver": "Image List Save",
    "ImageResize": "Image Resize",
    "ImageSize": "Image Size",
    "ConstrainImageNode": "Image Constrain",
    "DensePosePreprocessor": "DensePose Estimator",
    "DWPosePreprocessor": "DWPose Estimator",
    "AnimalPosePreprocessor": "AnimalPose Estimator",
    "MiDaSDepthPreprocessor": "MiDaS Depth Estimator"
}

WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]