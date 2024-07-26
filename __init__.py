from .nodes.load_node import LoadImageNode
from .nodes.save_node import SaveImageNode
from .nodes.save_node import SaveImagesNode
from .nodes.image_node import ImageResizeNode
from .nodes.image_node import ImageSizeNode


NODE_CLASS_MAPPINGS = {
    "ImageLoader": LoadImageNode,
    "ImageSaver": SaveImageNode,
    "ImagesSaver": SaveImagesNode,
    "ImageResize": ImageResizeNode,
    "ImageSize": ImageSizeNode,
}

