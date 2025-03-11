
import os
import torch
import cv2
import numpy as np
from PIL import Image
import folder_paths
from facefusion.modules.yoloface import YoloFace
from ..utils import tensor2pil, pil2tensor

class FaceCropNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.05}),
                "device": (['CPU', 'CUDA', 'CoreML', 'ROCM'], {"default": 'CPU'}),
            }
        }

    RETURN_TYPES = ("IMAGE","CROPINFO",)
    RETURN_NAMES = ("images", "crop_info",)
    FUNCTION = "crop"
    CATEGORY = "tbox/FaceFusion"

    def crop(self, images, device='CPU', weight=0.6):
        providers = ['CPUExecutionProvider']
        if device== 'CUDA':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == 'CoreML':
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        elif device == 'ROCM':
            providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
        
        yolo_path = folder_paths.get_full_path("facefusion", 'yoloface_8n.onnx')
        
        detector = YoloFace(model_path=yolo_path, providers=providers)
        
        crop_info = []
        for i, img in enumerate(images):
            pil = tensor2pil(img)
            image = np.ascontiguousarray(pil)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            face_list = detector.detect(image=image, conf=weight)
            crop_info.append(face_list)
        return (images, crop_info,)

    
