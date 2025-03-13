import os.path as osp
import torch
import numpy as np
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
from PIL import Image
from .base_cropper import Trajectory
from .human_cropper import HumanCropper

from liveportrait.modules.landmark_runner_animal import XPoseRunner as AnimalLandmarkRunner
from liveportrait.config.crop_config import CropConfig
from liveportrait.utils.crop import (
    average_bbox_lst,
    crop_image,
    crop_image_by_bbox,
    parse_bbox_from_landmark,
)

class AnimalCropper(HumanCropper):
    def __init__(self,  **kwargs) -> None:
        super().__init__(**kwargs)
        self.animal_landmark_runner = AnimalLandmarkRunner(
            model_config_path=self.crop_cfg.xpose_config_file_path,
            model_checkpoint_path=self.crop_cfg.xpose_ckpt_path,
            embeddings_cache_path=self.crop_cfg.xpose_embedding_cache_path,
            flag_use_half_precision=kwargs.get("flag_use_half_precision", True),
        )
        self.animal_landmark_runner.warmup()
    
    def crop_source(self, source_rgb_lst):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()
        tmp_dct = {
                'animal_face_9': 'animal_face',
                'animal_face_68': 'face'
            }
    
        for idx, frame_rgb in enumerate(source_rgb_lst):
            img_rgb_pil = Image.fromarray(frame_rgb)
            if idx == 0 or trajectory.start == -1:
                lmk = self.animal_landmark_runner.run(
                    img_rgb_pil,
                    'face',
                    tmp_dct[self.crop_cfg.animal_face_type],
                    0,
                    0
                )
                trajectory.start, trajectory.end = idx, idx
            else:
                # TODO: add IOU check for tracking
                lmk = self.animal_landmark_runner.run(
                    img_rgb_pil,
                    'face',
                    tmp_dct[self.crop_cfg.animal_face_type],
                    0,
                    0
                )
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)

            # crop the face
            ret_dct = crop_image(
                frame_rgb,  # ndarray
                lmk,  # 106x2 or Nx2
                dsize=self.crop_cfg.dsize,
                scale=self.crop_cfg.scale,
                vx_ratio=self.crop_cfg.vx_ratio,
                vy_ratio=self.crop_cfg.vy_ratio,
                flag_do_rot=self.crop_cfg.flag_do_rot,
            )

            # update a 256x256 version for network input
            ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
            ret_dct["lmk_crop_256x256"] = ret_dct["pt_crop"] * 256 / self.crop_cfg.dsize

            trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop_256x256"])
            trajectory.lmk_crop_lst.append(ret_dct["lmk_crop_256x256"])
            trajectory.M_c2o_lst.append(ret_dct['M_c2o'])

        return {
            "frame_crop_lst": trajectory.frame_rgb_crop_lst,
            "lmk_crop_lst": trajectory.lmk_crop_lst,
            "M_c2o_lst": trajectory.M_c2o_lst,
        }
    

if __name__ == '__main__':
    from liveportrait.utils.helper import draw_landmarks
    from liveportrait.utils.io import contiguous
    from liveportrait.utils.video import images2video

    def test_image(input_file, cropper) :
        image = cv2.imread(input_file)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = cropper.crop_source([image])
        dst = result['frame_crop_lst'][0]
        lmk = result['lmk_crop_lst'][0]
        frame = draw_landmarks(frame=dst, landmarks=lmk)
        cv2.imwrite(f'./output_crop2.jpg', frame)
    

    cropConfig = CropConfig()
    cropper = AnimalCropper(crop_cfg=cropConfig)
    test_image('../assets/cat1.jpg', cropper)
