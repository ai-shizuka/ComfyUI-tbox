import os.path as osp
import torch
import numpy as np
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

from .base_cropper import Trajectory

from liveportrait.modules.face_analysis_diy import FaceAnalysisDIY
from liveportrait.modules.landmark_runner_human import LandmarkRunner as HumanLandmarkRunner
from liveportrait.config.crop_config import CropConfig
from liveportrait.utils.io import contiguous
from liveportrait.utils.crop import (
    average_bbox_lst,
    crop_image,
    crop_image_by_bbox,
    parse_bbox_from_landmark,
)

class HumanCropper(object):
    def __init__(self,  **kwargs) -> None:
        self.device_id = kwargs.get("device_id", 0)
        self.providers = kwargs.get("providers", ["CPUExecutionProvider"])

        self.crop_cfg: CropConfig = kwargs.get("crop_cfg", None)
        self.face_analysis_wrapper = FaceAnalysisDIY(
                    name="buffalo_l",
                    root=self.crop_cfg.insightface_root,
                    providers=self.providers,
                )
        self.face_analysis_wrapper.prepare(ctx_id=self.device_id, det_size=(512, 512), det_thresh=self.crop_cfg.det_thresh)
        self.face_analysis_wrapper.warmup()
        
        self.human_landmark_runner = HumanLandmarkRunner(
            model_path=self.crop_cfg.landmark_ckpt_path,
            providers=self.providers
        )
        self.human_landmark_runner.warmup()
        
    def crop_source(self, source_rgb_lst):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()

        for idx, frame_rgb in enumerate(source_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=self.crop_cfg.direction,
                    max_face_num=self.crop_cfg.max_face_num,
                )
                if len(src_face) == 0:
                    print(f"No face detected in the frame #{idx}")
                    continue
                elif len(src_face) > 1:
                    print(f"More than one face detected in the source frame_{idx}, only pick one face by rule {direction}.")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                # TODO: add IOU check for tracking
                lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
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
    
    def crop_driving(self, driving_rgb_lst):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()

        for idx, frame_rgb in enumerate(driving_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=self.crop_cfg.direction,
                )
                if len(src_face) == 0:
                    print(f"No face detected in the frame #{idx}")
                    continue
                elif len(src_face) > 1:
                    print(f"More than one face detected in the driving frame_{idx}, only pick one face by rule {self.crop_cfg.direction}.")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)
            
            ret_bbox = parse_bbox_from_landmark(
                lmk,
                scale=self.crop_cfg.scale_crop_driving_video,
                vx_ratio_crop_driving_video=self.crop_cfg.vx_ratio_crop_driving_video,
                vy_ratio=self.crop_cfg.vy_ratio_crop_driving_video,
            )["bbox"]
            bbox = [
                ret_bbox[0, 0],
                ret_bbox[0, 1],
                ret_bbox[2, 0],
                ret_bbox[2, 1],
            ]  # 4,
            trajectory.bbox_lst.append(bbox)  # bbox
            trajectory.frame_rgb_lst.append(frame_rgb)

            # ret_dct = crop_image(
            #     frame_rgb,  # ndarray
            #     lmk,  # 106x2 or Nx2
            #     dsize=self.crop_cfg.dsize,
            #     scale=self.crop_cfg.scale,
            #     vx_ratio=self.crop_cfg.vx_ratio,
            #     vy_ratio=self.crop_cfg.vy_ratio,
            #     flag_do_rot=self.crop_cfg.flag_do_rot,
            # )

            # # update a 512x512 version for network input
            # ret_dct["img_crop_512x512"] = cv2.resize(ret_dct["img_crop"], (512, 512), interpolation=cv2.INTER_AREA)
            # ret_dct["lmk_crop_512x512"] = ret_dct["pt_crop"] * 512 / self.crop_cfg.dsize

            # trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop_512x512"])
            # trajectory.lmk_crop_lst.append(ret_dct["lmk_crop_512x512"])
        
        global_bbox = average_bbox_lst(trajectory.bbox_lst)

        for idx, (frame_rgb, lmk) in enumerate(zip(trajectory.frame_rgb_lst, trajectory.lmk_lst)):
            ret_dct = crop_image_by_bbox(
                frame_rgb,
                global_bbox,
                lmk=lmk,
                dsize=self.crop_cfg.dsize,
                flag_rot=False,
                borderValue=(0, 0, 0),
            )
            trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop"])
            trajectory.lmk_crop_lst.append(ret_dct["lmk_crop"])

        return {
            "frame_crop_lst": trajectory.frame_rgb_crop_lst,
            "lmk_crop_lst": trajectory.lmk_crop_lst,
        }




if __name__ == '__main__':
    from rich.progress import track
    from liveportrait.utils.video import images2video
    from liveportrait.utils.helper import draw_landmarks
    
    def test_image(input_file, cropper) :
        image = cv2.imread(input_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = cropper.crop_driving([image])
        dst = result['frame_crop_lst'][0]
        lmk = result['lmk_crop_lst'][0]
        frame = draw_landmarks(frame=dst, landmarks=lmk)
        cv2.imwrite(f'../output_crop.jpg', frame)
    

        
    def test_video(input_file, cropper) :
        cap = cv2.VideoCapture(input_file)
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或根据你的需要选择不同的编码器
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        for i in track(range(total), description='Read Video Frame....', transient=True):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        
        results = cropper.crop_driving(frames)
        frames = []

        for i in track(range(total), description='Draw Landmarks....', transient=True):
            dst = results['frame_crop_lst'][i]
            lmk = results['lmk_crop_lst'][i]
            frame = draw_landmarks(frame=dst, landmarks=lmk)
            frames.append(frame)

        images2video(images=frames, wfp='../output_crop.mp4', fps=fps, image_mode='bgr')
   
    input = '../assets/dzq.mp4'
    cropConfig = CropConfig()
    cropper = HumanCropper(crop_cfg=cropConfig, providers=["CUDAExecutionProvider"])
    test_video(input, cropper)
    