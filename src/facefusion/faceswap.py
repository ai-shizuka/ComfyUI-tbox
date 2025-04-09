import os
from dataclasses import dataclass, field
from typing import Literal, List, Tuple

from .modules.inswapper import InSwapper
from .modules.uniface import UniFace
from .modules.arcface import ArcFaceW600k
from .modules.gfpgan import GFPGAN
from .modules.yoloface import YoloFace
from .utils.affine import arcface_128_v2, ffhq_512, warp_face_by_landmark, paste_back, blend_frame
from .facemask import FaceMasker
from liveportrait.config.base_config import PrintableConfig



@dataclass(repr=False)  # use repr from PrintableConfig
class FaceSwapConfig(PrintableConfig):
    dsize: Tuple[int,int] = (256,256) # input shape
    model_name: Literal['inswapper_128', 'uniface_256'] = 'inswapper_128'
    model_path: str = ''
    providers:  List[str] = field(default_factory=lambda: ["CPUExecutionProvider"]) #Literal['CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'TensorrtExecutionProvider']
    gfpgan_blend: float = 0.8
    yoloface_weight: float = 0.6


class FaceSwapper(object):
    def __init__(self, config, mask_cfg):
        self.config = config
        if config.model_name == 'uniface_256':
            uniface_path = os.path.join(config.model_path, 'uniface_256.onnx')
            self.uniface = UniFace(model_path=uniface_path, providers=config.providers)
            
        elif config.model_name == 'inswapper_128':
            inswapper_path = os.path.join(config.model_path, 'inswapper_128.onnx')
            arcface_path = os.path.join(config.model_path, 'arcface_w600k_r50.onnx')
            self.inswapper = InSwapper(model_path=inswapper_path, providers=config.providers)
            self.recognizer = ArcFaceW600k(model_path=arcface_path, providers=config.providers)
        
        yolo_path = os.path.join(config.model_path, 'yoloface_8n.onnx')
        self.yolo = YoloFace(model_path=yolo_path, providers=config.providers)
        
        mask_cfg.model_path = config.model_path
        mask_cfg.providers = config.providers
        self.masker = FaceMasker(mask_cfg)
        
        gfpgan_path = os.path.join(config.model_path, 'gfpgan_1.4.onnx')
        self.gfpgan = GFPGAN(model_path=gfpgan_path, providers=config.providers)
        
    def execute(self, source, target, crop_info):
        face_list = self.yolo.detect(image=source, conf=self.config.yoloface_weight)
        if len(face_list) <= 0:
            raise RuntimeError("no source face detected!")

        if len(crop_info) == 0:
            return target
        
        if self.config.model_name == 'uniface_256':
            output = self.swap_uniface(source=source, source_crop_info=face_list[0], target=target, target_crop_info=crop_info[0])
        elif self.config.model_name == 'inswapper_128':
            output = self.swap_inswapper(source=source, source_crop_info=face_list[0], target=target, target_crop_info=crop_info[0])
        
        return output
    
    def swap_uniface(self, source, source_crop_info, target, target_crop_info):
        source_face, _ = warp_face_by_landmark(source, source_crop_info[1], ffhq_512, self.uniface.source_size)
        target_face, affine = warp_face_by_landmark(target, target_crop_info[1], ffhq_512, self.uniface.target_size)
        
        crop_mask = self.masker.create_mask(target_face)
        output = self.uniface.swap(source_face, target_face)        

        if self.config.gfpgan_blend >= 0.01:
            output = self.gfpgan.run(output)
        
        output = paste_back(target, output, crop_mask, affine)
        
        if self.config.gfpgan_blend >= 0.01:
            output = blend_frame(target, output, self.config.gfpgan_blend)
            
        return output
    
    def swap_inswapper(self, source, source_crop_info, target, target_crop_info):
        embedding = self.recognizer.embedding(image=source, landmarks=source_crop_info[1])
        target_face, affine = warp_face_by_landmark(target, target_crop_info[1], arcface_128_v2, self.config.dsize)
        
        crop_mask = self.masker.create_mask(target_face)
        output = self.inswapper.swap(embedding[0], target_face)
        
        if self.config.gfpgan_blend >= 0.01:
            output = self.gfpgan.run(output)
        
        output = paste_back(target, output, crop_mask, affine)
        
        if self.config.gfpgan_blend >= 0.01:
            output = blend_frame(target, output, self.config.gfpgan_blend)
            
        return output
             