from PIL import Image, ImageSequence, ImageOps
import torch
import requests
from io import BytesIO
import os
import numpy as np
import time

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
    FUNCTION = "save"
    CATEGORY = "tbox"
    OUTPUT_NODE = True
    
    def save(self, images, path, quality):
        format = os.path.splitext(path)[1][1:]
        image = images[0] 
        img = Image.fromarray((255. * image.cpu().numpy()).astype(np.uint8))
        save_image(img, path, format, quality)
        return {}
        
class SaveImagesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "path": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "prefix": ("STRING", {"default": "image"}),
                "format": (["PNG", "JPG", "WEBP", "BMP"],),
                "quality": ([100, 95, 90, 85, 80, 75, 70, 60, 50], {"default": 100}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "tbox"
    OUTPUT_NODE = True
    
    def save(self, images, path, prefix, format, quality):
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
    

# if __name__ == "__main__":
#     img, name = load_image("/Users/wadahana/workspace/AI/tbox.ai/data/tbox/task/20240704/50f524e9a28e63f9ecb5746f98353942/target.jpg")
#     img_out, mask_out = pil2tensor(img)
#     print(f'img_out shape: {img_out.shape}')
#     # images = (img_out,)
#     for image in img_out:
#         print(f'image shape: {image.shape}')
#         img1 = Image.fromarray((255. * image.cpu().numpy()).astype(np.uint8))
#     save_image(img1, "/Users/wadahana/workspace/AI/tbox.ai/data/tbox/task/20240704/50f524e9a28e63f9ecb5746f98353942/output11.png", "png", 90)

