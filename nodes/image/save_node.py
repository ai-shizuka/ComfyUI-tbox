import os
import time
import numpy as np
from PIL import Image, ImageSequence, ImageOps

#from load_node import load_image, pil2tensor

def save_image(img, filepath, format, quality):
    print(f"save_image >> path: {filepath}")
    print(f'save_image >> img: {img}')
    try:
        if format in ["jpg", "jpeg"]:
            img.convert("RGB").save(filepath, format="JPEG", quality=quality, subsampling=0)
        elif format == "webp":
            img.save(filepath, format="WEBP", quality=quality, method=6)
        elif format == "bmp":
            img.save(filepath, format="BMP")
        else:
            img.save(filepath, format="PNG", optimize=True)
    except Exception as e:
        print(f"Error saving {filepath}: {str(e)}")
        
class SaveImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "path": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "quality": ([100, 95, 90, 85, 80, 75, 70, 60, 50], {"default": 100}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save_image"
    CATEGORY = "tbox/Image"
    OUTPUT_NODE = True
    
    def save_image(self, images, path, quality):
        filepaht = path.split('\n')[0]
        format = os.path.splitext(filepaht)[1][1:]
        image = images[0] 
        img = Image.fromarray((255. * image.cpu().numpy()).astype(np.uint8))
        save_image(img, filepaht, format, quality)
        return {}
        
class SaveImagesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "path": ("STRING", {"multiline": False, "dynamicPrompts": False}),
                "prefix": ("STRING", {"default": "image"}),
                "format": (["PNG", "JPG", "WEBP", "BMP"],),
                "quality": ([100, 95, 90, 85, 80, 75, 70, 60, 50], {"default": 100}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save_image"
    CATEGORY = "tbox/Image"
    OUTPUT_NODE = True
    
    def save_image(self, images, path, prefix, format, quality):
        format = format.lower()            
        for i, image in enumerate(images):
            img = Image.fromarray((255. * image.cpu().numpy()).astype(np.uint8))
            filepath = self.generate_filename(path, prefix, i, format)
            save_image(img, filepath, format, quality)
        return {}

    def IS_CHANGED(s, images):
        return time.time()
    
    def generate_filename(self, save_dir, prefix, index, format):
        base_filename = f"{prefix}_{index+1}.{format}"
        filename = os.path.join(save_dir, base_filename)
        counter = 1
        while os.path.exists(filename):
            filename = os.path.join(save_dir, f"{prefix}_{index+1}_{counter}.{format}")
            counter += 1
        return filename
