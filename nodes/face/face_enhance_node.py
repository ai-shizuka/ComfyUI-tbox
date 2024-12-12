import os
import torch
import cv2
import numpy as np
from PIL import Image
from facefusion.gfpgan_onnx import GFPGANOnnx
from facefusion.yoloface_onnx import YoloFaceOnnx
from facefusion.affine import create_box_mask, warp_face_by_landmark, paste_back

import folder_paths
from ..utils import tensor2pil, pil2tensor
    
# class GFPGANProvider:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "model_name": ("IMAGE", ["gfpgan_1.4.onnx"]),
#             },
#         }
    
#     RETURN_TYPES = ("GFPGAN_MODEL",)
#     RETURN_NAMES = ("model",)
#     FUNCTION = "load_model"
#     CATEGORY = "tbox/facefusion"

#     def load_model(self, model_name):
#         return (model_name,)


class GFPGANNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": (['gfpgan_1.3', 'gfpgan_1.4'], {"default": 'gfpgan_1.4'}),
                "device": (['CPU', 'CUDA', 'CoreML', 'ROCM'], {"default": 'CPU'}),
                "weight": ("FLOAT", {"default": 0.8}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "process"
    CATEGORY = "tbox/facefusion"

    def process(self, images, model_name, device='CPU', weight=0.8):
        providers = ['CPUExecutionProvider']
        if device== 'CUDA':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == 'CoreML':
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        elif device == 'ROCM':
            providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
        
        gfpgan_path = folder_paths.get_full_path("facefusion", f'{model_name}.onnx')
        yolo_path = folder_paths.get_full_path("facefusion", 'yoloface_8n.onnx')
        
        detector = YoloFaceOnnx(model_path=yolo_path, providers=providers)
        enhancer = GFPGANOnnx(model_path=gfpgan_path, providers=providers)
        
        image_list = []
        for i, img in enumerate(images):
            pil = tensor2pil(img)
            image = np.ascontiguousarray(pil)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            output = image
            face_list = detector.detect(image=image, conf=0.7)
            for index, face in enumerate(face_list):
                cropped, affine_matrix = warp_face_by_landmark(image, face.landmarks, enhancer.input_size)
                box_mask = create_box_mask(enhancer.input_size, 0.3, (0,0,0,0))
                crop_mask = np.minimum.reduce([box_mask]).clip(0, 1)
                result = enhancer.run(cropped)
                output = paste_back(output, result, crop_mask, affine_matrix)
            image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(image)
            image_list.append(pil2tensor(pil))
        image_list = torch.stack([tensor.squeeze() for tensor in image_list])
        return (image_list,)

    
