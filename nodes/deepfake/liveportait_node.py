
import os
import torch
import cv2
import numpy as np
import folder_paths

from . import model_path
from ..utils import image_to_tensor, tensor_to_image, is_package_installed
from liveportrait.config.base_config import liveportrait_path
from liveportrait.config.crop_config import CropConfig
from liveportrait.config.inference_config import InferenceConfig
from liveportrait.human_cropper import HumanCropper
from liveportrait.human_pipeline import HumanPipeline


support_animal = is_package_installed("MultiScaleDeformableAttention")



def getOnnxProvidersFromDevice(device):
    providers = ['CPUExecutionProvider']
    if device == 'CUDA':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif device == 'CoreML':
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    elif device == 'ROCM':
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
    return providers

    
class LivePortraitSourceCropNode:
    @classmethod
    def INPUT_TYPES(cls):
        type_options = ['human']
        if support_animal:
            type_options.append('animal')
        return {
            "required": {
                "images": ("IMAGE",),
                "type": (type_options, {"default": 'human'}),
                "device": (['CPU', 'CUDA', 'CoreML', 'ROCM'], {"default": 'CPU'}),
            }
        }
    RETURN_TYPES = ('IMAGE', 'LP_CROPINFO', 'LP_TYPE')
    RETURN_NAMES = ('images', 'crop_info', 'type')
    FUNCTION = "crop"
    CATEGORY = "tbox/LivePortrait"
    
    def crop(self, images, type='human', device='CPU'):
        providers = getOnnxProvidersFromDevice(device)

        print(f'liveportrait_path: {liveportrait_path}')
        cropConfig = CropConfig()
        cropConfig.insightface_root = os.path.join(folder_paths.models_dir, "insightface")
        cropConfig.landmark_ckpt_path = os.path.join(folder_paths.models_dir, "liveportrait", 'landmark.onnx')
        cropConfig.xpose_config_file_path = os.path.abspath(os.path.join(liveportrait_path, "modules/XPose/config_model/UniPose_SwinT.py"))
        cropConfig.xpose_embedding_cache_path = os.path.abspath(os.path.join(liveportrait_path, 'resources/clip_embedding'))
        cropConfig.xpose_ckpt_path = os.path.abspath(os.path.join(folder_paths.models_dir, 'liveportrait/animal/xpose.pth'))
        
        #cropConfig.landmark_ckpt_path = folder_paths.get_full_path("liveportrait", 'landmark.onnx')
        frames = (images.cpu().numpy() * 255).astype(np.uint8) 
        if type == 'animal':
            from liveportrait.animal_cropper import AnimalCropper
            cropper = AnimalCropper(crop_cfg=cropConfig, providers=providers)
            crop_info = cropper.crop_source(frames)
        else:
            cropper = HumanCropper(crop_cfg=cropConfig, providers=providers)
            crop_info = cropper.crop_source(frames)

        return (images, crop_info, type)

class LivePortraitDrivingCropNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "device": (['CPU', 'CUDA', 'CoreML', 'ROCM'], {"default": 'CPU'}),
            }
        }

    RETURN_TYPES = ('IMAGE', 'LP_CROPINFO',)
    RETURN_NAMES = ('images', 'crop_info',)
    FUNCTION = "crop"
    CATEGORY = "tbox/LivePortrait"

    def crop(self, images, device='CPU'):
        providers = getOnnxProvidersFromDevice(device)
        liveportrait_path =  os.path.abspath('../../src/liveprotrait')
        print(f'liveportrait_path: {liveportrait_path}')
        
        cropConfig = CropConfig()
        cropConfig.insightface_root = os.path.join(folder_paths.models_dir, "insightface")
        cropConfig.landmark_ckpt_path = os.path.join(folder_paths.models_dir, "liveportrait", 'landmark.onnx')
        
        cropper = HumanCropper(crop_cfg=cropConfig, providers=providers)
    
        frames = (images.cpu().numpy() * 255).astype(np.uint8) 
        crop_info = cropper.crop_driving(frames)
        return (images, crop_info,)

class LivePortraitMotionNode:
    @classmethod
    def INPUT_TYPES(cls):
        type_options = ['human']
        if support_animal:
            type_options.append('animal')
        return {
            "required": {
                "source_lst": ("IMAGE",),
                "source_crop_info": ("LP_CROPINFO",),
                "driving_lst": ("IMAGE",),
                "driving_crop_info": ("LP_CROPINFO",),
                "type": ("LP_TYPE",),
                "frame_rate": ("INT,FLOAT", { "default": 25.0, "step": 1.0, "min": 1.0, "max": 60.0 }),
                "device": (['cpu', 'cuda', 'mps'], {"default": 'cpu'}),
            }
        }
    RETURN_TYPES = ("DRIVING_TEMPLATE",)
    RETURN_NAMES = ("driving_template",)
    FUNCTION = "calc_motion"
    CATEGORY = "tbox/LivePortrait"
    
    def calc_motion(self, source_lst, source_crop_info, driving_lst, driving_crop_info, type='human', frame_rate=25, device='cpu'):
  
        
        source_rgb_lst = (source_lst.cpu().numpy() * 255).astype(np.uint8) 
        source_rgb_lst = [source_rgb_lst[i] for i in range(source_rgb_lst.shape[0])]
        
        driving_rgb_lst = (driving_lst.cpu().numpy() * 255).astype(np.uint8) 
        driving_rgb_lst = [driving_rgb_lst[i] for i in range(driving_rgb_lst.shape[0])]
        inferConfig = InferenceConfig()
        inferConfig.device = device
        
        if type == 'animal':
            from liveportrait.animal_pipeline import AnimalPipeline
            pipeline = AnimalPipeline(inference_cfg=inferConfig)
        else:
            pipeline = HumanPipeline(inference_cfg=inferConfig)    
            
        driving_template = pipeline.calc_driving_template(
            fps=frame_rate, 
            source_rgb_lst=source_rgb_lst, 
            source_crop_info=source_crop_info, 
            driving_rgb_lst=driving_rgb_lst, 
            driving_crop_info=driving_crop_info)
        return (driving_template,)

class LivePortraitAnimateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_lst": ("IMAGE",),
                "source_crop_info": ("LP_CROPINFO",),
                "driving_template": ("DRIVING_TEMPLATE",),
                "frame_rate": ("INT,FLOAT", { "default": 25.0, "step": 1.0, "min": 1.0, "max": 60.0 }),
                "device": (['cpu', 'cuda', 'mps'], {"default": 'cpu'}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "animate"
    CATEGORY = "tbox/LivePortrait"
    
    def animate(self, source_lst, source_crop_info, driving_template, frame_rate, device='cpu'):
        inferConfig = InferenceConfig()
        inferConfig.device = device
        
        
        source_rgb_lst = (source_lst.cpu().numpy() * 255).astype(np.uint8) 
 
        if driving_template['type'] == 'animal' :
            from liveportrait.animal_pipeline import AnimalPipeline
            pipeline = AnimalPipeline(inference_cfg=inferConfig)
        else:
            pipeline = HumanPipeline(inference_cfg=inferConfig)
        result = pipeline.animate(
            fps=frame_rate, 
            source_rgb_lst=source_rgb_lst, 
            source_crop_info=source_crop_info, 
            driving_template=driving_template)
        
        images = torch.stack([image_to_tensor(image).squeeze() for image in result])
        return (images,)
    


# if animal_supported == True:
   
    
#     class LivePortraitAnimalMotionNode:
#         @classmethod
#         def INPUT_TYPES(cls):
#             return {
#                 "required": {
#                     "source_lst": ("IMAGE",),
#                     "source_crop_info": ("LP_CROPINFO",),
#                     "driving_lst": ("IMAGE",),
#                     "driving_crop_info": ("LP_CROPINFO",),
#                     "frame_rate": ("INT,FLOAT", { "default": 25.0, "step": 1.0, "min": 1.0, "max": 60.0 }),
#                     "device": (['cpu', 'cuda', 'mps'], {"default": 'cpu'}),
#                 }
#             }
#         RETURN_TYPES = ("DRIVING_TEMPLATE",)
#         RETURN_NAMES = ("driving_template",)
#         FUNCTION = "calc_motion"
#         CATEGORY = "tbox/LivePortrait"
        
#         def calc_motion(self, source_lst, source_crop_info, driving_lst, driving_crop_info, frame_rate, device='cpu'):
#             inferConfig = InferenceConfig()
#             inferConfig.device = device
#             pipeline = AnimalPipeline(inference_cfg=inferConfig)
            
#             source_rgb_lst = (source_lst.cpu().numpy() * 255).astype(np.uint8) 
#             source_rgb_lst = [source_rgb_lst[i] for i in range(source_rgb_lst.shape[0])]
            
#             driving_rgb_lst = (driving_lst.cpu().numpy() * 255).astype(np.uint8) 
#             driving_rgb_lst = [driving_rgb_lst[i] for i in range(driving_rgb_lst.shape[0])]

#             driving_template = pipeline.calc_driving_template(
#                 fps=frame_rate, 
#                 source_rgb_lst=source_rgb_lst, 
#                 source_crop_info=source_crop_info, 
#                 driving_rgb_lst=driving_rgb_lst, 
#                 driving_crop_info=driving_crop_info)
#             return (driving_template,)
        
#     class LivePortraitAnimalAnimateNode:
#         @classmethod
#         def INPUT_TYPES(cls):
#             return {
#                 "required": {
#                     "source_lst": ("IMAGE",),
#                     "source_crop_info": ("LP_CROPINFO",),
#                     "driving_template": ("DRIVING_TEMPLATE",),
#                     "frame_rate": ("INT,FLOAT", { "default": 25.0, "step": 1.0, "min": 1.0, "max": 60.0 }),
#                     "device": (['cpu', 'cuda', 'mps'], {"default": 'cpu'}),
#                 }
#             }
#         #"gfpgan_blend": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05, "tooltip": "0: disable gfpgan",}),
#         RETURN_TYPES = ("IMAGE",)
#         RETURN_NAMES = ("images",)
#         FUNCTION = "animate"
#         CATEGORY = "tbox/LivePortrait"
        
#         def animate(self, source_lst, source_crop_info, driving_template, frame_rate, device='cpu'):
#             inferConfig = InferenceConfig()
#             inferConfig.device = device
#             pipeline = HumanPipeline(inference_cfg=inferConfig)
            
#             source_rgb_lst = (source_lst.cpu().numpy() * 255).astype(np.uint8) 
    
#             result = pipeline.animate(
#                 fps=frame_rate, 
#                 source_rgb_lst=source_rgb_lst, 
#                 source_crop_info=source_crop_info, 
#                 driving_template=driving_template)
            
#             images = torch.stack([image_to_tensor(image).squeeze() for image in result])
#             return (images,)