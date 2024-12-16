from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management
import nodes

class LineArt_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            coarse=(["disable", "enable"], {"default": "enable"}),
            resolution=("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 64})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "tbox/ControlNet Preprocessors"
    

    def execute(self, image, resolution=512, **kwargs):
        from lineart import LineartDetector

        model = LineartDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, coarse = kwargs["coarse"] == "enable")
        del model
        return (out, )
    
class Lineart_Standard_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            guassian_sigma=("FLOAT", {"default":6.0, "max": 100.0}),
            intensity_threshold=("INT", {"default": 8, "max": 16}),
            resolution=("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 64})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "tbox/ControlNet Preprocessors"
    

    def execute(self, image, guassian_sigma=6, intensity_threshold=8, resolution=512, **kwargs):
        from lineart import LineartStandardDetector
        return (common_annotator_call(LineartStandardDetector(), image, guassian_sigma=guassian_sigma, intensity_threshold=intensity_threshold, resolution=resolution), )