
from PIL import Image, ImageSequence, ImageOps
import torch
import requests
import nodes
from io import BytesIO
import os
import comfy.utils
import numpy as np

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
    CATEGORY = "tbox"
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
    CATEGORY = "tbox"
    
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
    