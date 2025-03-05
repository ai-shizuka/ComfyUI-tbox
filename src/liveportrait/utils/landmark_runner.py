# coding: utf-8

import os.path as osp
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import torch
import pickle
import numpy as np
import onnxruntime
from PIL import Image
from torchvision.ops import nms

from liveportrait.utils.timer import Timer
from liveportrait.utils.crop import crop_image, _transform_pts
from liveportrait.modules.XPose import transforms as T
from liveportrait.modules.XPose.models import build_model
from liveportrait.modules.XPose.predefined_keypoints import *
from liveportrait.modules.XPose.util import box_ops
from liveportrait.modules.XPose.util.config import Config
from liveportrait.modules.XPose.util.misc import clean_state_dict

def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


def to_ndarray(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return np.array(obj)


class HumanLandmarkRunner(object):
    """landmark runner"""

    def __init__(self, **kwargs):
        ckpt_path = kwargs.get('ckpt_path')
        onnx_provider = kwargs.get('onnx_provider', 'cuda')  # 默认用cuda
        device_id = kwargs.get('device_id', 0)
        self.dsize = kwargs.get('dsize', 224)
        self.timer = Timer()

        if onnx_provider.lower() == 'cuda':
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=[
                    ('CUDAExecutionProvider', {'device_id': device_id})
                ]
            )
        elif onnx_provider.lower() == 'mps':
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=[
                    'CoreMLExecutionProvider'
                ]
            )
        else:
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = 4  # 默认线程数为 4
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=['CPUExecutionProvider'],
                sess_options=opts
            )

    def _run(self, inp):
        out = self.session.run(None, {'input': inp})
        return out

    def run(self, img_rgb: np.ndarray, lmk=None):
        if lmk is not None:
            crop_dct = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
            img_crop_rgb = crop_dct['img_crop']
        else:
            # NOTE: force resize to 224x224, NOT RECOMMEND!
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            scale = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {
                'M_c2o': np.array([
                    [scale, 0., 0.],
                    [0., scale, 0.],
                    [0., 0., 1.],
                ], dtype=np.float32),
            }

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)

        out_lst = self._run(inp)
        out_pts = out_lst[2]

        # 2d landmarks 203 points
        lmk = to_ndarray(out_pts[0]).reshape(-1, 2) * self.dsize  # scale to 0-224
        lmk = _transform_pts(lmk, M=crop_dct['M_c2o'])

        return lmk

    def warmup(self):
        self.timer.tic()

        dummy_image = np.zeros((1, 3, self.dsize, self.dsize), dtype=np.float32)

        _ = self._run(dummy_image)

        elapse = self.timer.toc()
        print(f'LandmarkRunner warmup time: {elapse:.3f}s')




class XPoseRunner(object):
    def __init__(self, model_config_path, model_checkpoint_path, embeddings_cache_path=None, cpu_only=False, **kwargs):
        self.device_id = kwargs.get("device_id", 0)
        self.flag_use_half_precision = kwargs.get("flag_use_half_precision", True)
        self.device = f"cuda:{self.device_id}" if not cpu_only else "cpu"
        self.model = self.load_animal_model(model_config_path, model_checkpoint_path, self.device)
        self.timer = Timer()
        # Load cached embeddings if available
        print(f'embeddings_cache_path: {embeddings_cache_path}')
        try:
            with open(f'{embeddings_cache_path}_9.pkl', 'rb') as f:
                self.ins_text_embeddings_9, self.kpt_text_embeddings_9 = pickle.load(f)
            with open(f'{embeddings_cache_path}_68.pkl', 'rb') as f:
                self.ins_text_embeddings_68, self.kpt_text_embeddings_68 = pickle.load(f)
            print("Loaded cached embeddings from file.")
        except Exception:
            raise ValueError("Could not load clip embeddings from file, please check your file path.")

    def load_animal_model(self, model_config_path, model_checkpoint_path, device):
        args = Config.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def load_image(self, input_image):
        image_pil = input_image.convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),  # NOTE: fixed size to 800
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image

    def get_unipose_output(self, image, instance_text_prompt, keypoint_text_prompt, box_threshold, IoU_threshold):
        instance_list = instance_text_prompt.split(',')

        if len(keypoint_text_prompt) == 9:
            # torch.Size([1, 512]) torch.Size([9, 512])
            ins_text_embeddings, kpt_text_embeddings = self.ins_text_embeddings_9, self.kpt_text_embeddings_9
        elif len(keypoint_text_prompt) ==68:
            # torch.Size([1, 512]) torch.Size([68, 512])
            ins_text_embeddings, kpt_text_embeddings = self.ins_text_embeddings_68, self.kpt_text_embeddings_68
        else:
            raise ValueError("Invalid number of keypoint embeddings.")
        target = {
            "instance_text_prompt": instance_list,
            "keypoint_text_prompt": keypoint_text_prompt,
            "object_embeddings_text": ins_text_embeddings.float(),
            "kpts_embeddings_text": torch.cat((kpt_text_embeddings.float(), torch.zeros(100 - kpt_text_embeddings.shape[0], 512, device=self.device)), dim=0),
            "kpt_vis_text": torch.cat((torch.ones(kpt_text_embeddings.shape[0], device=self.device), torch.zeros(100 - kpt_text_embeddings.shape[0], device=self.device)), dim=0)
        }

        self.model = self.model.to(self.device)
        image = image.to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=self.flag_use_half_precision):
                outputs = self.model(image[None], [target])

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]
        keypoints = outputs["pred_keypoints"][0][:, :2 * len(keypoint_text_prompt)]

        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        keypoints_filt = keypoints.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        keypoints_filt = keypoints_filt[filt_mask]

        keep_indices = nms(box_ops.box_cxcywh_to_xyxy(boxes_filt), logits_filt.max(dim=1)[0], iou_threshold=IoU_threshold)

        filtered_boxes = boxes_filt[keep_indices]
        filtered_keypoints = keypoints_filt[keep_indices]

        return filtered_boxes, filtered_keypoints

    def run(self, input_image, instance_text_prompt, keypoint_text_example, box_threshold, IoU_threshold):
        if keypoint_text_example in globals():
            keypoint_dict = globals()[keypoint_text_example]
        elif instance_text_prompt in globals():
            keypoint_dict = globals()[instance_text_prompt]
        else:
            keypoint_dict = globals()["animal"]

        keypoint_text_prompt = keypoint_dict.get("keypoints")
        keypoint_skeleton = keypoint_dict.get("skeleton")

        image_pil, image = self.load_image(input_image)
        boxes_filt, keypoints_filt = self.get_unipose_output(image, instance_text_prompt, keypoint_text_prompt, box_threshold, IoU_threshold)

        size = image_pil.size
        H, W = size[1], size[0]
        keypoints_filt = keypoints_filt[0].squeeze(0)
        kp = np.array(keypoints_filt.cpu())
        num_kpts = len(keypoint_text_prompt)
        Z = kp[:num_kpts * 2] * np.array([W, H] * num_kpts)
        Z = Z.reshape(num_kpts * 2)
        x = Z[0::2]
        y = Z[1::2]
        return np.stack((x, y), axis=1)

    def warmup(self):
        self.timer.tic()

        img_rgb = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        self.run(img_rgb, 'face', 'face', box_threshold=0.0, IoU_threshold=0.0)

        elapse = self.timer.toc()
        print(f'XPoseRunner warmup time: {elapse:.3f}s')


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