
import os
import torch
import cv2
import numpy as np
import folder_paths
from PIL import Image
from typing import Literal, List, get_args
from .__init__ import model_path
from ..utils import tensor_to_image, image_to_tensor
from facefusion.utils.affine import ffhq_512, warp_face_by_landmark, paste_back
from facefusion.utils.mask import create_bbox_mask, FaceMaskRegion, FaceMaskRegionMap, FaceMaskAllRegion
from facefusion.faceswap import FaceSwapConfig, FaceSwapper
from facefusion.facemask import FaceMaskConfig

        
class FaceMaskConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BOOLEAN", {"default": True, "label": "enable bbox "}),
                "occlusion": ("BOOLEAN", {"default": False, "label": "enable occlusion"}),
                "region": ("BOOLEAN", {"default": False, "label": "enable region "}),
                "bbox_blur": ("FLOAT", {"default": 0.3, "min": 0, "max": 1, "step": 0.1, "tooltip": "",}),
                "region_list": ("STRING", {"multiline": True,
                    "default": "skin, left-eyebrow, right-eyebrow, left-eye, right-eye, glasses, nose, mouth, upper-lip, lower-lip",
                    "tooltip": "skin, left-eyebrow, right-eyebrow, left-eye, right-eye, glasses, nose, mouth, upper-lip, lower-lip",
                }),
            },
        }
    
    RETURN_TYPES = ("MASKCFG",)
    RETURN_NAMES = ("mask_cfg",)
    FUNCTION = "process"
    CATEGORY = "tbox/FaceFusion"
            
    def process(self, bbox=True, occlusion=False, region=False, bbox_blur=0.3, region_list=[]):
        if region == False:
            region_list = []
        else :
            region_list = self.parse_region_string(region_list)
            
        mask_cfg = FaceMaskConfig()
        mask_cfg.bbox = bbox
        mask_cfg.bbox_blur = bbox_blur
        mask_cfg.bbox_padding = [0,0,0,0]
        mask_cfg.occlusion = occlusion
        mask_cfg.region = region
        mask_cfg.region_list = region_list 
        
        return (mask_cfg, )
        
    def parse_region_string(self, input_str: str) -> List[FaceMaskRegion]:
        valid_regions = set(get_args(FaceMaskRegion))
        result = []
        for item in input_str.split(','):
            cleaned = item.strip()
            if cleaned in valid_regions:
                result.append(cleaned)
        return result

                
class FaceSwapNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "targets": ("IMAGE",),
                "crop_info": ("CROPINFO",),
                "mask_cfg": ("MASKCFG",),
                "model_name": (['inswapper_128', 'uniface_256'], {"default": 'inswapper_128'}),
                "device": (['CPU', 'CUDA', 'CoreML', 'ROCM'], {"default": 'CPU'}),
                "yoloface_weight": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.05, "tooltip": "",}),
                "gfpgan_blend": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05, "tooltip": "0: disable gfpgan",}),
            }
        }

    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "tbox/FaceFusion"

    def process(self, source, targets, crop_info, mask_cfg, model_name='inswapper_128', device='CPU', gfpgan_blend=0.8, yoloface_weight=0.6):
        providers = ['CPUExecutionProvider']
        if device== 'CUDA':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == 'CoreML':
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        elif device == 'ROCM':
            providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']

        print(f'mask_cfg: {mask_cfg}')
        
        fs_cfg = FaceSwapConfig() 
        fs_cfg.providers = providers
        fs_cfg.model_name = model_name
        fs_cfg.model_path = model_path
        fs_cfg.dsize = (256,256)
        fs_cfg.gfpgan_blend = gfpgan_blend
        fs_cfg.yoloface_weight = yoloface_weight
        
        swapper = FaceSwapper(fs_cfg, mask_cfg)
        
        print(f'length of crop_info: {len(crop_info)}')
        print(f'shape of targets: {targets.shape}')
 
        if len(crop_info) != targets.shape[0]:
            raise ValueError("size of crop_info not equal to target images")
        if source.shape[0] < 1:
            raise ValueError("no source image")
        
        source = tensor_to_image(source[0]) 
        image_list = []
        for i, target in enumerate(targets):
            image = tensor_to_image(target)
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            output = swapper.execute(source, image, crop_info[i])
            image_list.append(image_to_tensor(output))
        image_list = torch.stack([tensor.squeeze() for tensor in image_list])
        return (image_list,)            
    
