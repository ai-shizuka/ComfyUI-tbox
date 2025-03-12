import os
import cv2
import os.path as osp
import torch
import safetensors.torch
from collections import OrderedDict
import numpy as np
from scipy.spatial import ConvexHull # pylint: disable=E0401,E0611
from typing import Union

from liveportrait.modules.spade_generator import SPADEDecoder
from liveportrait.modules.warping_network import WarpingNetwork
from liveportrait.modules.motion_extractor import MotionExtractor
from liveportrait.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from liveportrait.modules.stitching_retargeting_network import StitchingRetargetingNetwork


def tensor_to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """transform torch.Tensor into numpy.ndarray"""
    if isinstance(data, torch.Tensor):
        return data.data.cpu().numpy()
    return data

def calc_motion_multiplier(
    kp_source: Union[np.ndarray, torch.Tensor],
    kp_driving_initial: Union[np.ndarray, torch.Tensor]
) -> float:
    """calculate motion_multiplier based on the source image and the first driving frame"""
    kp_source_np = tensor_to_numpy(kp_source)
    kp_driving_initial_np = tensor_to_numpy(kp_driving_initial)

    source_area = ConvexHull(kp_source_np.squeeze(0)).volume
    driving_area = ConvexHull(kp_driving_initial_np.squeeze(0)).volume
    motion_multiplier = np.sqrt(source_area) / np.sqrt(driving_area)
    # motion_multiplier = np.cbrt(source_area) / np.cbrt(driving_area)

    return motion_multiplier


def suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind(".")
    if pos == -1:
        return ""
    return filename[pos + 1:]


def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]


def basename(filename):
    """a/b/c.jpg -> c"""
    return prefix(osp.basename(filename))


def remove_suffix(filepath):
    """a/b/c.jpg -> a/b/c"""
    return osp.join(osp.dirname(filepath), basename(filepath))


def is_image(file_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return file_path.lower().endswith(image_extensions)


def is_video(file_path):
    if file_path.lower().endswith((".mp4", ".mov", ".avi", ".webm")) or osp.isdir(file_path):
        return True
    return False


def is_template(file_path):
    if file_path.endswith(".pkl"):
        return True
    return False


def mkdir(d, log=False):
    # return self-assined `d`, for one line code
    if not osp.exists(d):
        os.makedirs(d, exist_ok=True)
        if log:
            print(f"Make dir: {d}")
    return d


def squeeze_tensor_to_numpy(tensor):
    out = tensor.data.squeeze(0).cpu().numpy()
    return out


def dct2device(dct: dict, device):
    for key in dct:
        if isinstance(dct[key], torch.Tensor):
            dct[key] = dct[key].to(device)
        else:
            dct[key] = torch.tensor(dct[key]).to(device)
    return dct


def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat

def filter_checkpoint_for_model(checkpoint, prefix):
    """Filter and adjust the checkpoint dictionary for a specific model based on the prefix."""
    # Create a new dictionary where keys are adjusted by removing the prefix and the model name
    filtered_checkpoint = {
        key.replace(prefix + "_module.", ""): value
        for key, value in checkpoint.items()
        if key.startswith(prefix)
    }
    return filtered_checkpoint


def load_model(ckpt_path, model_config, device, model_type):
    model_params = model_config['model_params'][f'{model_type}_params']

    if model_type == 'appearance_feature_extractor':
        model = AppearanceFeatureExtractor(**model_params).to(device)
    elif model_type == 'motion_extractor':
        model = MotionExtractor(**model_params).to(device)
    elif model_type == 'warping_module':
        model = WarpingNetwork(**model_params).to(device)
    elif model_type == 'spade_generator':
        model = SPADEDecoder(**model_params).to(device)
    elif model_type == 'stitching_retargeting_module':
        # Special handling for stitching and retargeting module
        config = model_config['model_params']['stitching_retargeting_module_params']
        if ckpt_path.lower().endswith(".safetensors"):
            checkpoint = safetensors.torch.load_file(ckpt_path, device=device)
        else:
            checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        #checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
        stitcher.load_state_dict(filter_checkpoint_for_model(checkpoint, 'retarget_shoulder'))
#        stitcher.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
        stitcher = stitcher.to(device)
        stitcher.eval()

        retargetor_lip = StitchingRetargetingNetwork(**config.get('lip'))
        retargetor_lip.load_state_dict(filter_checkpoint_for_model(checkpoint, 'retarget_mouth'))
        #retargetor_lip.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_mouth']))
        retargetor_lip = retargetor_lip.to(device)
        retargetor_lip.eval()

        retargetor_eye = StitchingRetargetingNetwork(**config.get('eye'))
        retargetor_eye.load_state_dict(filter_checkpoint_for_model(checkpoint, 'retarget_eye'))
        #retargetor_eye.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_eye']))
        retargetor_eye = retargetor_eye.to(device)
        retargetor_eye.eval()

        return {
            'stitching': stitcher,
            'lip': retargetor_lip,
            'eye': retargetor_eye
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if ckpt_path.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt_path, device=device)
    else:
        sd = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
#    model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
    model.load_state_dict(sd)
    model.eval()
    return model


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

def draw_landmarks(frame, landmarks, color=(0, 255, 0), radius=2, thickness=-1):
    """
    在帧上绘制 203 个面部关键点。
    
    :param frame: 要绘制的图像 (numpy array)
    :param landmarks: 包含 203 个 (x, y) 坐标的列表或 numpy 数组
    :param color: 点的颜色，默认为绿色 (BGR)
    :param radius: 点的半径
    :param thickness: 线条厚度，-1 表示填充点
    """
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), radius, color, thickness)
    return frame