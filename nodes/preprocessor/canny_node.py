from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management
import nodes

class Canny_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            low_threshold=("INT", {"default": 100, "min": 0, "max": 255}),
            high_threshold=("INT", {"default": 100, "min": 0, "max": 255}),
            resolution=("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 64})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "tbox/ControlNet Preprocessors"

    def execute(self, image, low_threshold=100, high_threshold=200, resolution=512, **kwargs):
        from canny import CannyDetector

        return (common_annotator_call(CannyDetector(), image, low_threshold=low_threshold, high_threshold=high_threshold, resolution=resolution), )
