from .nodes.load_node import LoadImageNode
from .nodes.save_node import SaveImageNode
from .nodes.save_node import SaveImagesNode



NODE_CLASS_MAPPINGS = {
    "ImageLoader": LoadImageNode,
    "ImageSaver": SaveImageNode,
    "ImagesSaver": SaveImagesNode,
}

