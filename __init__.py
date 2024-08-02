import sys
from pathlib import Path
from .utils import here

sys.path.insert(0, str(Path(here, "src").resolve()))

from .nodes.load_node import LoadImageNode
from .nodes.save_node import SaveImageNode
from .nodes.save_node import SaveImagesNode
from .nodes.image_node import ImageResizeNode
from .nodes.image_node import ImageSizeNode
from .nodes.dwpose_node import DWPose_Preprocessor, AnimalPose_Preprocessor

sys.path.insert(0, str(Path(here, "src").resolve()))

print(f'here>> {here} ')

NODE_CLASS_MAPPINGS = {
    "ImageLoader": LoadImageNode,
    "ImageSaver": SaveImageNode,
    "ImagesSaver": SaveImagesNode,
    "ImageResize": ImageResizeNode,
    "ImageSize": ImageSizeNode,
    "DWPosePreprocessor": DWPose_Preprocessor,
    "AnimalPosePreprocessor": AnimalPose_Preprocessor,
}


NODE_DISPLAY_NAME_MAPPINGS = {
     "ImageLoader": "ImageLoader",
    "ImageSaver": "ImageSaver",
    "ImagesSaver": "ImagesSaver",
    "ImageResize": "ImageResizer",
    "ImageSize": "ImageSize",
    "DWPosePreprocessor": "DWPose Estimator",
    "AnimalPosePreprocessor": "AnimalPose Estimator"
}