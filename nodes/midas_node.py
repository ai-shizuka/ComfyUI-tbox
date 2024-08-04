from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management
import numpy as np

class MIDAS_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            a =  ("FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 5.0, "step": 0.05}),
            bg_threshold = ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.05})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "tbox/ControlNet Preprocessors"

    def execute(self, image, a, bg_threshold, resolution=512, **kwargs):
        from midas import MidasDetector

        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_depth2image.py
        model = MidasDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, a=a, bg_th=bg_threshold)
        del model
        return (out, )
