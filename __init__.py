import sys
from pathlib import Path
from .utils import here

sys.path.insert(0, str(Path(here, "src").resolve()))

from .nodes.image.load_node import LoadImageNode
from .nodes.image.save_node import SaveImageNode
from .nodes.image.save_node import SaveImagesNode
from .nodes.image.size_node import ImageResizeNode
from .nodes.image.size_node import ImageSizeNode
from .nodes.image.size_node import ConstrainImageNode
from .nodes.video.load_node import LoadVideoNode
from .nodes.video.save_node import SaveVideoNode
from .nodes.video.info_node import VideoInfoNode

from .nodes.preprocessor.midas_node import MIDAS_Depth_Map_Preprocessor
from .nodes.preprocessor.dwpose_node import DWPose_Preprocessor, AnimalPose_Preprocessor
from .nodes.preprocessor.densepose_node import DensePose_Preprocessor
LoadVideoNode

NODE_CLASS_MAPPINGS = {
    "ImageLoader": LoadImageNode,
    "ImageSaver": SaveImageNode,
    "ImagesSaver": SaveImagesNode,
    "ImageResize": ImageResizeNode,
    "ImageSize": ImageSizeNode,
    "VideoLoader": LoadVideoNode,
    "VideoSaver": SaveVideoNode,
    "VideoInfo": VideoInfoNode,
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
    "VideoLoader": "Video Load",
    "VideoSaver": "Video Save",
    "VideoInfo": "Video Info", 
    "ConstrainImageNode": "Image Constrain",
    "DensePosePreprocessor": "DensePose Estimator",
    "DWPosePreprocessor": "DWPose Estimator",
    "AnimalPosePreprocessor": "AnimalPose Estimator",
    "MiDaSDepthPreprocessor": "MiDaS Depth Estimator"
}

WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]