

import torch
import comfy.utils
import numpy as np
from PIL import Image, ImageSequence, ImageOps

class ConstrainImageNode:
    """
    A node that constrains an image to a maximum and minimum size while maintaining aspect ratio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_width": ("INT", {"default": 1024, "min": 0}),
                "max_height": ("INT", {"default": 1024, "min": 0}),
                "min_width": ("INT", {"default": 0, "min": 0}),
                "min_height": ("INT", {"default": 0, "min": 0}),
                "crop_if_required": (["yes", "no"], {"default": "no"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "constrain_image"
    CATEGORY = "tbox/Image"
    OUTPUT_IS_LIST = (True,)

    def constrain_image(self, images, max_width, max_height, min_width, min_height, crop_if_required):
        crop_if_required = crop_if_required == "yes"
        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")

            current_width, current_height = img.size
            aspect_ratio = current_width / current_height

            constrained_width = max(min(current_width, min_width), max_width)
            constrained_height = max(min(current_height, min_height), max_height)

            if constrained_width / constrained_height > aspect_ratio:
                constrained_width = max(int(constrained_height * aspect_ratio), min_width)
                if crop_if_required:
                    constrained_height = int(current_height / (current_width / constrained_width))
            else:
                constrained_height = max(int(constrained_width / aspect_ratio), min_height)
                if crop_if_required:
                    constrained_width = int(current_width / (current_height / constrained_height))

            resized_image = img.resize((constrained_width, constrained_height), Image.LANCZOS)

            if crop_if_required and (constrained_width > max_width or constrained_height > max_height):
                left = max((constrained_width - max_width) // 2, 0)
                top = max((constrained_height - max_height) // 2, 0)
                right = min(constrained_width, max_width) + left
                bottom = min(constrained_height, max_height) + top
                resized_image = resized_image.crop((left, top, right, bottom))

            resized_image = np.array(resized_image).astype(np.float32) / 255.0
            resized_image = torch.from_numpy(resized_image)[None,]
            results.append(resized_image)
                
        return (results,)
    
# https://github.com/bronkula/comfyui-fitsize
class ImageSizeNode: 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "count")
    FUNCTION = "get_size"
    CATEGORY = "tbox/Image"
    def get_size(self, image):
        print(f'shape of image:{image.shape}')
        return (image.shape[2], image.shape[1], image[0])



class ImageResizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "method": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
            },
        "optional": {
                "width": ("INT,FLOAT", { "default": 0.0, "step": 0.1 }),
                "height": ("INT,FLOAT", { "default": 0.0, "step": 0.1 }),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "resize"
    CATEGORY = "tbox/Image"
    
    def resize(self, image, method, width, height):
        print(f'shape of image:{image.shape}, resolution:{width}x{height} type: {type(width)}, {type(height)}')
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1,1)
            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = comfy.utils.common_upscale(samples, width, height, method, True)
            s = s.movedim(1,-1)
        return (s,)
    