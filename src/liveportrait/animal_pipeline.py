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
from liveportrait.utils.camera import get_rotation_matrix
from liveportrait.utils.video import images2video#, concat_frames, get_fps, add_audio_to_video, has_audio_stream, video2gif
from liveportrait.utils.crop import _transform_img, prepare_paste_back, paste_back
from liveportrait.utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from liveportrait.utils.helper import basename, dct2device, is_video, is_template, remove_suffix, is_image, calc_motion_multiplier

# from .utils.viz import viz_lmk
from liveportrait.wrapper import AnimalWrapper


class AnimalPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig):
        self.live_portrait_wrapper_animal: AnimalWrapper = AnimalWrapper(inference_cfg=inference_cfg)
    
    def make_motion_template(self, I_lst, **kwargs):
        n_frames = I_lst.shape[0]
        template_dct = {
            'type': 'animal',
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
        }

        for i in range(n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            x_i_info = self.live_portrait_wrapper_animal.get_kp_info(I_i)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

        return template_dct
    
    def calc_driving_template(self, fps, source_rgb_lst, source_crop_info, driving_rgb_lst, driving_crop_info):
        ######## process driving info ########
        flag_is_source_video = False
        flag_is_driving_video = True
        
        ######## make motion template ########
        if source_crop_info == None or source_crop_info == None or driving_rgb_lst == None or driving_crop_info == None:
            raise Exception(f"not source or driving files!")
        print("Start making driving motion template...")
        driving_n_frames = len(driving_rgb_lst)
        source_n_frames = len(source_rgb_lst)
        if flag_is_source_video and flag_is_driving_video:
            n_frames = min(source_n_frames, driving_n_frames)  # minimum number as the number of the animated frames
            driving_rgb_lst = driving_rgb_lst[:n_frames]
            driving_crop_info = driving_crop_info[:n_frames]
        elif flag_is_source_video and not flag_is_driving_video:
            n_frames = source_n_frames
        else:
            n_frames = driving_n_frames
            
        driving_rgb_crop_lst, driving_lmk_crop_lst = driving_crop_info['frame_crop_lst'], driving_crop_info['lmk_crop_lst']
        driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
        #######################################
    
        I_d_lst = self.live_portrait_wrapper_animal.prepare_videos(driving_rgb_crop_256x256_lst)
        driving_template_dct = self.make_motion_template(I_d_lst, output_fps=fps)
        return driving_template_dct
    
    def animate(self, fps, source_rgb_lst, source_crop_info, driving_template):
        ######## process driving info ########
        flag_is_source_video = False
        flag_is_driving_video = True
        driving_template_dct = driving_template
        n_frames = driving_template_dct['n_frames']

        # set output_fps
        output_fps = driving_template_dct.get('output_fps', fps)
        print(f'The FPS of template: {output_fps}')
        
        return self.do_execute(source_rgb_lst, source_crop_info, n_frames, driving_template_dct)
    
    def do_execute(self, source_rgb_lst, source_crop_info, n_frames, driving_template_dct): 
        inf_cfg = self.live_portrait_wrapper_animal.inference_cfg
        device = self.live_portrait_wrapper_animal.device
        
        flag_is_source_video = False
        flag_is_driving_video = True
        
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            I_p_pstbk_lst = []
            print("Prepared pasteback mask done.")
            
        ######## process source info ########
        img_rgb = source_rgb_lst[0]    
        img_crop_256x256 = source_crop_info['frame_crop_lst'][0]
            
        I_s = self.live_portrait_wrapper_animal.prepare_source(img_crop_256x256)
        x_s_info = self.live_portrait_wrapper_animal.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper_animal.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper_animal.transform_keypoint(x_s_info)

        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, source_crop_info['M_c2o_lst'][0], dsize=(img_rgb.shape[1], img_rgb.shape[0]))

        ######## animate ########
        I_p_lst = []
        for i in range(n_frames):

            x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)

            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys
            delta_new = x_d_i_info['exp']
            t_new = x_d_i_info['t']
            t_new[..., 2].fill_(0)  # zero tz
            scale_new = x_s_info['scale']

            x_d_i = scale_new * (x_c_s @ R_d_i + delta_new) + t_new

            if i == 0:
                x_d_0 = x_d_i
                motion_multiplier = calc_motion_multiplier(x_s, x_d_0)

            x_d_diff = (x_d_i - x_d_0) * motion_multiplier
            x_d_i = x_d_diff + x_s

            if not inf_cfg.flag_stitching:
                pass
            else:
                x_d_i = self.live_portrait_wrapper_animal.stitching(x_s, x_d_i)

            x_d_i = x_s + (x_d_i - x_s) * inf_cfg.driving_multiplier
            out = self.live_portrait_wrapper_animal.warp_decode(f_s, x_s, x_d_i)
            I_p_i = self.live_portrait_wrapper_animal.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                I_p_pstbk = paste_back(I_p_i, source_crop_info['M_c2o_lst'][0], img_rgb, mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)
            
        return I_p_pstbk_lst
        

if __name__ == '__main__':
    from liveportrait.animal_cropper import AnimalCropper
    image_input = "../assets/cat.jpg"
    #image_input = "./assets/liuyifei.jpeg"
    video_input = '../assets/liveportrait.mp4'
    video_output =  '../output.mp4'
    cap = cv2.VideoCapture(video_input)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cropConfig = CropConfig()
    inferConfig = InferenceConfig()
    inferConfig.device = 'cuda'
    cropper = AnimalCropper(crop_cfg=cropConfig, providers=["CUDAExecutionProvider"])
    pipeline = AnimalPipeline(inference_cfg=inferConfig)
    
    frames = []
    for i in track(range(total), description='Read Video Frame....', transient=True):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    image = cv2.imread(image_input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    source_crop_info = cropper.crop_source([image])
    driving_crop_info = cropper.crop_driving(frames)
    
    driving_template = pipeline.calc_driving_template(source_rgb_lst=[image], source_crop_info=source_crop_info, driving_rgb_lst=frames, driving_crop_info=driving_crop_info, fps=fps)
   
    result = pipeline.animate(fps=fps, source_rgb_lst=[image], source_crop_info=source_crop_info, driving_template=driving_template)
   
    print(f'shape of result: {len(result)},')

    images2video(images=result, wfp=video_output, fps=fps)
