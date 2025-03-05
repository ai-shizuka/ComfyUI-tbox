# coding: utf-8

"""
Pipeline of LivePortrait (Animal)
"""

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
from rich.progress import track


from liveportrait.config.inference_config import InferenceConfig
from liveportrait.config.crop_config import CropConfig
from liveportrait.utils.cropper import AnimalCropper
from liveportrait.utils.camera import get_rotation_matrix
from liveportrait.utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream, video2gif
from liveportrait.utils.crop import _transform_img, prepare_paste_back, paste_back
from liveportrait.utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from liveportrait.utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, calc_motion_multiplier

# from .utils.viz import viz_lmk
from liveportrait.wrapper import AnimalWrapper


class AnimalPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper_animal: AnimalWrapper = AnimalWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg, image_type='animal_face', flag_use_half_precision=inference_cfg.flag_use_half_precision)
