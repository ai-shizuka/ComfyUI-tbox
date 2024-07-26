from PIL import Image, ImageSequence, ImageOps
import torch
import requests
from io import BytesIO
import os
import numpy as np


def pil2tensor(img):
    output_images = []
    output_masks = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)


def load_image(image_source):
    if image_source.startswith('http'):
        print(image_source)
        response = requests.get(image_source)
        img = Image.open(BytesIO(response.content))
        file_name = image_source.split('/')[-1]
    else:
        img = Image.open(image_source)
        file_name = os.path.basename(image_source)
    return img, file_name


class LoadImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"multiline": True, "dynamicPrompts": False})
            }
        }


    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load"
    CATEGORY = "tbox"

    def load(self, path):
        print(path)
        img, name = load_image(path)
        img_out, mask_out = pil2tensor(img)
        return (img_out, mask_out)


if __name__ == "__main__":
    img, name = load_image("https://creativestorage.blob.core.chinacloudapi.cn/test/bird.png")
    img_out, mask_out = pil2tensor(img)

