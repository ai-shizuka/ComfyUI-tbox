
import torch
import numpy as np
from PIL import Image, ImageSequence, ImageOps
from ..utils import tensor2pil, pil2tensor

PADDING = 4

class WatermarkNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "logo_list": ("IMAGE",),
            },
            "optional": {
                "logo_mask": ("MASK",),
                "enabled": ("BOOLEAN", {"default": True}),}
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "watermark"
    CATEGORY = "tbox/Image"
    
    def watermark(self, images, logo_list, logo_mask, enabled):
        outputs = []
        if enabled == False: 
            return(images,)
        print(f'logo shape: {logo_list.shape}')
        print(f'images shape: {images.shape}')
        logo = tensor2pil(logo_list[0]) 
        if logo_mask is not None:
            logo_mask = tensor2pil(logo_mask)
        for i, image in enumerate(images):
            img = tensor2pil(image) #Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            dst = self.add_watermark2(img, logo, logo_mask, 85)
            result = pil2tensor(dst)
            outputs.append(result)
        base_image = torch.stack([tensor.squeeze() for tensor in outputs])
        return (base_image,)
    
    def add_watermark2(self, image, logo, logo_mask, opacity=None):
        logo_width, logo_height = logo.size
        image_width, image_height = image.size
        if image_height <= logo_height + PADDING * 2 or image_width <= logo_width + PADDING * 2:
            return image
        y = image_height - logo_height - PADDING * 1
        x = PADDING
        print(f'logo size: {logo.size}')
        print(f'image size: {image.size}')
        logo = logo.convert('RGBA')
        opacity = int(opacity / 100 * 255)
        logo.putalpha(Image.new("L", logo.size, opacity))
        if logo_mask is not None:
            logo.putalpha(ImageOps.invert(logo_mask))

        position = (x, y)
        image.paste(logo, position, logo)
        return image
