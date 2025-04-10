
import os
import numpy as np
from PIL import Image
from wd14tagger import defaults, known_models, get_installed_models, wait_for_async, tag
from ..utils import tensor2pil

class WD14Tagger:
    @classmethod
    def INPUT_TYPES(s):
        extra = [name for name, _ in (os.path.splitext(m) for m in get_installed_models()) if name not in known_models]
        models = known_models + extra
        return {"required": {
            "images": ("IMAGE", ),
            "model": (models, { "default": defaults["model"] }),
            "device": (['CPU', 'CUDA', 'CoreML', 'ROCM'], {"default": 'CPU'}),
            "threshold": ("FLOAT", {"default": defaults["threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "character_threshold": ("FLOAT", {"default": defaults["character_threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "replace_underscore": ("BOOLEAN", {"default": defaults["replace_underscore"]}),
            "trailing_comma": ("BOOLEAN", {"default": defaults["trailing_comma"]}),
            "exclude_tags": ("STRING", {"default": defaults["exclude_tags"]}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    OUTPUT_NODE = True

    CATEGORY = f"tbox/Prompt"

    def process(self, images, model, device, threshold, character_threshold, exclude_tags="", replace_underscore=False, trailing_comma=False):
        providers = ['CPUExecutionProvider']
        if device== 'CUDA':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == 'CoreML':
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        elif device == 'ROCM':
            providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
        
        #pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags = ''
        for i, img in enumerate(images):
            image = tensor2pil(img) 
            tags = wait_for_async(lambda: tag(image, model, providers, threshold, character_threshold, exclude_tags, replace_underscore, trailing_comma))
            #tags.append()
        #    pbar.update(1)
        return (tags,)
        #return {"ui": {"tags": tags}, "result": (tags,)}

