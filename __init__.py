import sys
from pathlib import Path
from .utils import here
import platform

sys.path.insert(0, str(Path(here, "src").resolve()))

from .nodes.image.load_node import LoadImageNode
from .nodes.image.save_node import SaveImageNode
from .nodes.image.save_node import SaveImagesNode
from .nodes.image.size_node import ImageResizeNode
from .nodes.image.size_node import ImageSizeNode
from .nodes.image.size_node import ConstrainImageNode
from .nodes.image.watermark_node import WatermarkNode
from .nodes.mask.mask_node import MaskAddNode
from .nodes.video.load_node import LoadVideoNode
from .nodes.video.save_node import SaveVideoNode
from .nodes.video.info_node import VideoInfoNode
from .nodes.video.batch_node import BatchManagerNode
from .nodes.preprocessor.canny_node import Canny_Preprocessor
from .nodes.preprocessor.lineart_node import LineArt_Preprocessor
from .nodes.preprocessor.lineart_node import Lineart_Standard_Preprocessor
from .nodes.preprocessor.midas_node import MIDAS_Depth_Map_Preprocessor
from .nodes.preprocessor.dwpose_node import DWPose_Preprocessor, AnimalPose_Preprocessor
from .nodes.preprocessor.densepose_node import DensePose_Preprocessor
from .nodes.face.face_enhance_node import GFPGANNode
from .nodes.other.vram_node import PurgeVRAMNode

NODE_CLASS_MAPPINGS = {
    "PurgeVRAMNode": PurgeVRAMNode,
    "GFPGANNode": GFPGANNode,
    "MaskAddNode": MaskAddNode,
    "ImageLoader": LoadImageNode,
    "ImageSaver": SaveImageNode,
    "ImagesSaver": SaveImagesNode,
    "ImageResize": ImageResizeNode,
    "ImageSize": ImageSizeNode,
    "WatermarkNode": WatermarkNode,
    "VideoLoader": LoadVideoNode,
    "VideoSaver": SaveVideoNode,
    "VideoInfo": VideoInfoNode,
    "BatchManager": BatchManagerNode,
    "ConstrainImageNode": ConstrainImageNode,
    "DensePosePreprocessor": DensePose_Preprocessor,
    "DWPosePreprocessor": DWPose_Preprocessor,
    "AnimalPosePreprocessor": AnimalPose_Preprocessor,
    "MiDaSDepthPreprocessor": MIDAS_Depth_Map_Preprocessor,
    "CannyPreprocessor": Canny_Preprocessor,
    "LineArtPreprocessor": LineArt_Preprocessor,
    "LineartStandardPreprocessor": Lineart_Standard_Preprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PurgeVRAMNode":"PurgeVRAMNode",
    "GFPGANNode": "GFPGANNode",
    "MaskAddNode": "MaskAddNode",
    "ImageLoader": "Image Load",
    "ImageSaver": "Image Save",
    "ImagesSaver": "Image List Save",
    "ImageResize": "Image Resize",
    "ImageSize": "Image Size",
    "WatermarkNode": "Watermark",
    "VideoLoader": "Video Load",
    "VideoSaver": "Video Save",
    "VideoInfo": "Video Info", 
    "BatchManager": "Batch Manager",
    "ConstrainImageNode": "Image Constrain",
    "DensePosePreprocessor": "DensePose Estimator",
    "DWPosePreprocessor": "DWPose Estimator",
    "AnimalPosePreprocessor": "AnimalPose Estimator",
    "MiDaSDepthPreprocessor": "MiDaS Depth Estimator",
    "CannyPreprocessor": "Canny Edge Estimator",
    "LineArtPreprocessor": "Realistic Lineart",
    "LineartStandardPreprocessor": "Standard Lineart",
}


if platform.system() == "Darwin":
    WEB_DIRECTORY = "./web"
    __all__ = ["WEB_DIRECTORY", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
else:
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]