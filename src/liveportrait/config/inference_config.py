# coding: utf-8

"""
config dataclass used for inference
"""

import cv2
import os
from numpy import ndarray
import pickle as pkl
from dataclasses import dataclass, field
from typing import Literal, Tuple

from dataclasses import dataclass
from liveportrait.config.base_config import PrintableConfig, models_path, liveportrait_path

def load_lip_array():
    p = os.path.abspath(os.path.join(liveportrait_path, 'resources/lip_array.pkl'))
    with open(p, 'rb') as f:
        return pkl.load(f)
    
    
@dataclass(repr=False)  # use repr from PrintableConfig
class InferenceConfig(PrintableConfig):
  
    models_config: str = os.path.abspath(os.path.join(liveportrait_path,'config/models.yaml'))  # portrait animation config
    checkpoint_F: str = os.path.abspath(os.path.join(models_path, 'liveportrait/appearance_feature_extractor.safetensors'))  # path to checkpoint of F
    checkpoint_M: str = os.path.abspath(os.path.join(models_path,'liveportrait/motion_extractor.safetensors'))  # path to checkpoint pf M
    checkpoint_G: str = os.path.abspath(os.path.join(models_path,'liveportrait/spade_generator.safetensors'))  # path to checkpoint of G
    checkpoint_W: str = os.path.abspath(os.path.join(models_path,'liveportrait/warping_module.safetensors'))  # path to checkpoint of W
    checkpoint_S: str = os.path.abspath(os.path.join(models_path,'liveportrait/stitching_retargeting_module.safetensors'))  # path to checkpoint to S and R_eyes, R_lip

    # ANIMAL MODEL CONFIG, NOT EXPORTED PARAMS
    version_animals = "" # old version
    #version_animals = "_v1.1" # new (v1.1) version
    checkpoint_F_animal: str = os.path.abspath(os.path.join(models_path,f'liveportrait/animal/appearance_feature_extractor.safetensors'))  # path to checkpoint of F
    checkpoint_M_animal: str = os.path.abspath(os.path.join(models_path,f'liveportrait/animal/motion_extractor.safetensors'))  # path to checkpoint pf M
    checkpoint_G_animal: str = os.path.abspath(os.path.join(models_path,f'liveportrait/animal/spade_generator.safetensors'))  # path to checkpoint of G
    checkpoint_W_animal: str = os.path.abspath(os.path.join(models_path,f'liveportrait/animal/warping_module.safetensors'))  # path to checkpoint of W
    checkpoint_S_animal: str = os.path.abspath(os.path.join(models_path,f'liveportrait/animal/stitching_retargeting_module.safetensors'))  # path to checkpoint to S and R_eyes, R_lip, NOTE: use human temporarily!
    
    # EXPORTED PARAMS
    flag_use_half_precision: bool = True
    flag_crop_driving_video: bool = False
    device_id: int = 0
    device: str = "cpu"
    flag_normalize_lip: bool = True
    flag_source_video_eye_retargeting: bool = False
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_stitching: bool = True
    flag_relative_motion: bool = True
    flag_pasteback: bool = True
    flag_do_crop: bool = True
    flag_do_rot: bool = True
    flag_force_cpu: bool = False
    flag_do_torch_compile: bool = False
    driving_option: str = "pose-friendly" # "expression-friendly" or "pose-friendly"
    driving_multiplier: float = 1.0
    driving_smooth_observation_variance: float = 3e-7 # smooth strength scalar for the animated video when the input is a source video, the larger the number, the smoother the animated video; too much smoothness would result in loss of motion accuracy
    source_max_dim: int = 1280 # the max dim of height and width of source image or video
    source_division: int = 2 # make sure the height and width of source image or video can be divided by this number
    animation_region: Literal["exp", "pose", "lip", "eyes", "all"] = "all" # the region where the animation was performed, "exp" means the expression, "pose" means the head pose

    # NOT EXPORTED PARAMS
    lip_normalize_threshold: float = 0.03 # threshold for flag_normalize_lip
    source_video_eye_retargeting_threshold: float = 0.18 # threshold for eyes retargeting if the input is a source video
    anchor_frame: int = 0 # TO IMPLEMENT

    input_shape: Tuple[int, int] = (256, 256)  # input shape
    output_format: Literal['mp4', 'gif'] = 'mp4'  # output video format
    crf: int = 15  # crf for output video
    output_fps: int = 25 # default output fps

    mask_crop: ndarray = field(default_factory=lambda: cv2.imread(os.path.abspath(os.path.join(liveportrait_path, 'resources/mask_template.png')), cv2.IMREAD_COLOR))
    lip_array: ndarray = field(default_factory=load_lip_array)
    size_gif: int = 256 # default gif size, TO IMPLEMENT

