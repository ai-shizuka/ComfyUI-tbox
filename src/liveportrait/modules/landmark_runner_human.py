
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import torch
import numpy as np
import onnxruntime
from PIL import Image
from torchvision.ops import nms
from liveportrait.utils.timer import Timer
from liveportrait.utils.crop import crop_image, _transform_pts

def to_ndarray(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return np.array(obj)


class LandmarkRunner(object):
    """landmark runner"""

    def __init__(self, model_path, providers, dsize=224):
        self.dsize = dsize
        self.timer = Timer()

        self.session = onnxruntime.InferenceSession(
            model_path, providers=providers
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

