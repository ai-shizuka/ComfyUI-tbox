# coding: utf-8

"""
Pipeline of LivePortrait (Human)
"""

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
from rich.progress import track

#from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from liveportrait.utils.camera import get_rotation_matrix
from liveportrait.utils.video import images2video ## concat_frames, get_fps, add_audio_to_video, has_audio_stream
from liveportrait.utils.crop import prepare_paste_back, paste_back
from liveportrait.utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from liveportrait.utils.helper import basename, dct2device, is_video, is_template, remove_suffix, is_image, calc_motion_multiplier
from liveportrait.utils.filter import smooth

# from .utils.viz import viz_lmk
from liveportrait.wrapper import HumanWrapper


class HumanPipeline(object):
    def __init__(self, inference_cfg: InferenceConfig):
        self.live_portrait_wrapper: HumanWrapper = HumanWrapper(inference_cfg=inference_cfg)
    
    def make_motion_template(self, fps, I_lst, c_eyes_lst, c_lip_lst):
        n_frames = I_lst.shape[0]
        template_dct = {
            'type': 'human',
            'n_frames': n_frames,
            'output_fps': fps,
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        for i in range(n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            x_i_info = self.live_portrait_wrapper.get_kp_info(I_i)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_i_info)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

        return template_dct
    
    def calc_driving_template(self, fps, source_rgb_lst, source_crop_info, driving_rgb_lst, driving_crop_info):
        
        ######## process driving info ########
        flag_is_source_video = False
        flag_is_driving_video = True
        
        ######## make motion template ########
        if source_rgb_lst == None or source_crop_info == None or driving_rgb_lst == None or driving_crop_info == None:
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
    
        c_d_eyes_lst, c_d_lip_lst = self.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst)
        I_d_lst = self.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
        driving_template_dct = self.make_motion_template(fps=fps, I_lst=I_d_lst, c_eyes_lst=c_d_eyes_lst, c_lip_lst=c_d_lip_lst)
        return driving_template_dct
    
    def animate(self, fps, source_rgb_lst, source_crop_info, driving_template):
        ######## process driving info ########
        flag_is_source_video = False
        flag_is_driving_video = True
        
        driving_template_dct = driving_template
        c_d_eyes_lst = driving_template_dct['c_eyes_lst'] if 'c_eyes_lst' in driving_template_dct.keys() else driving_template_dct['c_d_eyes_lst'] # compatible with previous keys
        c_d_lip_lst = driving_template_dct['c_lip_lst'] if 'c_lip_lst' in driving_template_dct.keys() else driving_template_dct['c_d_lip_lst']
        driving_n_frames = driving_template_dct['n_frames']
        flag_is_driving_video = True if driving_n_frames > 1 else False
        if flag_is_source_video and flag_is_driving_video:
            n_frames = min(len(source_rgb_lst), driving_n_frames)  # minimum number as the number of the animated frames
        elif flag_is_source_video and not flag_is_driving_video:
            n_frames = len(source_rgb_lst)
        else:
            n_frames = driving_n_frames

        # set output_fps
        fps = driving_template_dct.get('output_fps', fps)
        print(f'The FPS of template: {fps}')
        
        return self.do_execute(source_rgb_lst, source_crop_info, n_frames, c_d_eyes_lst, c_d_lip_lst, driving_template_dct)
     
    def do_execute(self, source_rgb_lst, source_crop_info, n_frames, c_d_eyes_lst, c_d_lip_lst, driving_template_dct): 
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device = self.live_portrait_wrapper.device
        
        flag_is_source_video = False
        flag_is_driving_video = True
        
        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            I_p_pstbk_lst = []
            print("Prepared pasteback mask done.")
        
        I_p_lst = []
        R_d_0, x_d_0_info = None, None
        flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
        flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
        lip_delta_before_animation, eye_delta_before_animation = None, None
        

        ######## process source info ########
        if flag_is_source_video:
            print(f"Start making source motion template...")

            source_rgb_lst = source_rgb_lst[:n_frames]
            print(f'Source video is cropped, {len(source_crop_info["frame_crop_lst"])} frames are processed.')
            if len(source_crop_info["frame_crop_lst"]) is not n_frames:
                n_frames = min(n_frames, len(source_crop_info["frame_crop_lst"]))
            img_crop_256x256_lst, source_lmk_crop_lst, source_M_c2o_lst = source_crop_info['frame_crop_lst'], source_crop_info['lmk_crop_lst'], source_crop_info['M_c2o_lst']
            
            # if inf_cfg.flag_do_crop:
            #     ret_s = self.cropper.crop_source_video(source_rgb_lst, crop_cfg)
            #     print(f'Source video is cropped, {len(ret_s["frame_crop_lst"])} frames are processed.')
            #     if len(ret_s["frame_crop_lst"]) is not n_frames:
            #         n_frames = min(n_frames, len(ret_s["frame_crop_lst"]))
            #     img_crop_256x256_lst, source_lmk_crop_lst, source_M_c2o_lst = ret_s['frame_crop_lst'], ret_s['lmk_crop_lst'], ret_s['M_c2o_lst']
            # else:
            #     source_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(source_rgb_lst)
            #     img_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in source_rgb_lst]  # force to resize to 256x256

            c_s_eyes_lst, c_s_lip_lst = self.live_portrait_wrapper.calc_ratio(source_lmk_crop_lst)
            # save the motion template
            I_s_lst = self.live_portrait_wrapper.prepare_videos(img_crop_256x256_lst)
            source_template_dct = self.make_motion_template(I_s_lst, c_s_eyes_lst, c_s_lip_lst, output_fps=fps)

            key_r = 'R' if 'R' in driving_template_dct['motion'][0].keys() else 'R_d'  # compatible with previous keys
            if inf_cfg.flag_relative_motion:
                if flag_is_driving_video:
                    x_d_exp_lst = [source_template_dct['motion'][i]['exp'] + driving_template_dct['motion'][i]['exp'] - driving_template_dct['motion'][0]['exp'] for i in range(n_frames)]
                    x_d_exp_lst_smooth = smooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, device, inf_cfg.driving_smooth_observation_variance)
                else:
                    x_d_exp_lst = [source_template_dct['motion'][i]['exp'] + (driving_template_dct['motion'][0]['exp'] - inf_cfg.lip_array) for i in range(n_frames)]
                    x_d_exp_lst_smooth = [torch.tensor(x_d_exp[0], dtype=torch.float32, device=device) for x_d_exp in x_d_exp_lst]
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    if flag_is_driving_video:
                        x_d_r_lst = [(np.dot(driving_template_dct['motion'][i][key_r], driving_template_dct['motion'][0][key_r].transpose(0, 2, 1))) @ source_template_dct['motion'][i]['R'] for i in range(n_frames)]
                        x_d_r_lst_smooth = smooth(x_d_r_lst, source_template_dct['motion'][0]['R'].shape, device, inf_cfg.driving_smooth_observation_variance)
                    else:
                        x_d_r_lst = [source_template_dct['motion'][i]['R'] for i in range(n_frames)]
                        x_d_r_lst_smooth = [torch.tensor(x_d_r[0], dtype=torch.float32, device=device) for x_d_r in x_d_r_lst]
            else:
                if flag_is_driving_video:
                    x_d_exp_lst = [driving_template_dct['motion'][i]['exp'] for i in range(n_frames)]
                    x_d_exp_lst_smooth = smooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, device, inf_cfg.driving_smooth_observation_variance)
                else:
                    x_d_exp_lst = [driving_template_dct['motion'][0]['exp']]
                    x_d_exp_lst_smooth = [torch.tensor(x_d_exp[0], dtype=torch.float32, device=device) for x_d_exp in x_d_exp_lst]*n_frames
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    if flag_is_driving_video:
                        x_d_r_lst = [driving_template_dct['motion'][i][key_r] for i in range(n_frames)]
                        x_d_r_lst_smooth = smooth(x_d_r_lst, source_template_dct['motion'][0]['R'].shape, device, inf_cfg.driving_smooth_observation_variance)
                    else:
                        x_d_r_lst = [driving_template_dct['motion'][0][key_r]]
                        x_d_r_lst_smooth = [torch.tensor(x_d_r[0], dtype=torch.float32, device=device) for x_d_r in x_d_r_lst]*n_frames
        else:
            #source_lmk = source_crop_info['lmk_crop']
            #img_crop_256x256 = source_crop_info['img_crop_256x256']
            source_lmk = source_crop_info['lmk_crop_lst'][0]
            img_crop_256x256 = source_crop_info['frame_crop_lst'][0]
            
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_c_s = x_s_info['kp']
            R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

            # let lip-open scalar to be 0 at first
            if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
                c_d_lip_before_animation = [0.]
                combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                    lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, source_crop_info['M_c2o_lst'][0], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))
        
        ######## animate ########
        if flag_is_driving_video or (flag_is_source_video and not flag_is_driving_video):
            print(f"The animated video consists of {n_frames} frames.")
        else:
            print(f"The output of image-driven portrait animation is an image.")
        
        for i in range(n_frames):
            if flag_is_source_video:  # source video
                x_s_info = source_template_dct['motion'][i]
                x_s_info = dct2device(x_s_info, device)

                source_lmk = source_lmk_crop_lst[i]
                img_crop_256x256 = img_crop_256x256_lst[i]
                I_s = I_s_lst[i]
                f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)

                x_c_s = x_s_info['kp']
                R_s = x_s_info['R']
                x_s =x_s_info['x_s']

                # let lip-open scalar to be 0 at first if the input is a video
                if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
                    c_d_lip_before_animation = [0.]
                    combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                    if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                        lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)
                    else:
                        lip_delta_before_animation = None

                # let eye-open scalar to be the same as the first frame if the latter is eye-open state
                if flag_source_video_eye_retargeting and source_lmk is not None:
                    if i == 0:
                        combined_eye_ratio_tensor_frame_zero = c_s_eyes_lst[0]
                        c_d_eye_before_animation_frame_zero = [[combined_eye_ratio_tensor_frame_zero[0][:2].mean()]]
                        if c_d_eye_before_animation_frame_zero[0][0] < inf_cfg.source_video_eye_retargeting_threshold:
                            c_d_eye_before_animation_frame_zero = [[0.39]]
                    combined_eye_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eye_before_animation_frame_zero, source_lmk)
                    eye_delta_before_animation = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor_before_animation)

                if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:  # prepare for paste back
                    mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, source_M_c2o_lst[i], dsize=(source_rgb_lst[i].shape[1], source_rgb_lst[i].shape[0]))
            
            if flag_is_source_video and not flag_is_driving_video:
                x_d_i_info = driving_template_dct['motion'][0]
            else:
                x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)
            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys

            if i == 0:  # cache the first frame
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info.copy()
            
            delta_new = x_s_info['exp'].clone()
            if inf_cfg.flag_relative_motion:
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    # line
                    R_new = x_d_r_lst_smooth[i] if flag_is_source_video else (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                else:
                    R_new = R_s
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                    if flag_is_source_video:
                        for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                            delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :]
                        delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1]
                        delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2]
                        delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2]
                        delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:]
                    else:
                        if flag_is_driving_video:
                            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                        else:
                            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device))
                elif inf_cfg.animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        if flag_is_source_video:
                            delta_new[:, lip_idx, :] = x_d_exp_lst_smooth[i][lip_idx, :]
                        elif flag_is_driving_video:
                            delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, lip_idx, :]
                        else:
                            delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device)))[:, lip_idx, :]
                elif inf_cfg.animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        if flag_is_source_video:
                            delta_new[:, eyes_idx, :] = x_d_exp_lst_smooth[i][eyes_idx, :]
                        elif flag_is_driving_video:
                            delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, eyes_idx, :]
                        else:
                            delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
                if inf_cfg.animation_region == "all":
                    scale_new = x_s_info['scale'] if flag_is_source_video else x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                else:
                    scale_new = x_s_info['scale']
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    t_new = x_s_info['t'] if flag_is_source_video else x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    t_new = x_s_info['t']
            else:
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    R_new = x_d_r_lst_smooth[i] if flag_is_source_video else R_d_i
                else:
                    R_new = R_s
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                    for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                        delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :] if flag_is_source_video else x_d_i_info['exp'][:, idx, :]
                    delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1] if flag_is_source_video else x_d_i_info['exp'][:, 3:5, 1]
                    delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2] if flag_is_source_video else x_d_i_info['exp'][:, 5, 2]
                    delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2] if flag_is_source_video else x_d_i_info['exp'][:, 8, 2]
                    delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:] if flag_is_source_video else x_d_i_info['exp'][:, 9, 1:]
                elif inf_cfg.animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_exp_lst_smooth[i][lip_idx, :] if flag_is_source_video else x_d_i_info['exp'][:, lip_idx, :]
                elif inf_cfg.animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_exp_lst_smooth[i][eyes_idx, :] if flag_is_source_video else x_d_i_info['exp'][:, eyes_idx, :]
                scale_new = x_s_info['scale']
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    t_new = x_d_i_info['t']
                else:
                    t_new = x_s_info['t']
                    
            t_new[..., 2].fill_(0)  # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
            
            if inf_cfg.flag_relative_motion and inf_cfg.driving_option == "expression-friendly" and not flag_is_source_video and flag_is_driving_video:
                if i == 0:
                    x_d_0_new = x_d_i_new
                    motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
                    # motion_multiplier *= inf_cfg.driving_multiplier
                x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
                x_d_i_new = x_d_diff + x_s

            # Algorithm 1:
            if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # without stitching or retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new += lip_delta_before_animation
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
                else:
                    pass
            elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
            else:
                eyes_delta, lip_delta = None, None
                if inf_cfg.flag_eye_retargeting and source_lmk is not None:
                    c_d_eyes_i = c_d_eyes_lst[i]
                    combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                if inf_cfg.flag_lip_retargeting and source_lmk is not None:
                    c_d_lip_i = c_d_lip_lst[i]
                    combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

                if inf_cfg.flag_relative_motion:  # use x_s
                    x_d_i_new = x_s + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)

                if inf_cfg.flag_stitching:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
            
            x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
                if flag_is_source_video:
                    I_p_pstbk = paste_back(I_p_i, source_M_c2o_lst[i], source_rgb_lst[i], mask_ori_float)
                else:
                    I_p_pstbk = paste_back(I_p_i, source_crop_info['M_c2o_lst'][0], source_rgb_lst[0], mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)        
        
        return I_p_pstbk_lst
            


if __name__ == '__main__':
    from liveportrait.human_cropper import HumanCropper
    from liveportrait.utils.helper import draw_landmarks
    from rich.progress import track
    from datetime import datetime
    
    image_input = "../assets/ami.jpg"
    #image_input = "../assets/liuyifei.jpg"
    video_input = '../assets/dzq.mp4'
    video_output =  '../output_live.mp4'
    cap = cv2.VideoCapture(video_input)

    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    cropConfig = CropConfig()
    inferConfig = InferenceConfig()
    cropper = HumanCropper(crop_cfg=cropConfig, providers=["CPUExecutionProvider"])
    pipeline = HumanPipeline(inference_cfg=inferConfig)
    
    print(f'{datetime.now()} read video frames start.')
       
    frames = []
    for i in track(range(total), description='Read Video Frame....', transient=True):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f'{datetime.now()} read video frames finished.')
    
    print(f'frames.shape: {frames.shape}')

    
    image = cv2.imread(image_input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    source_crop_info = cropper.crop_source([image])
    print(f'{datetime.now()} >>>> crop source image finished.')
    
    driving_crop_info = cropper.crop_driving(frames)
    
    print(f'{datetime.now()} >>>> crop driving frames finished.')
    
    driving_template = pipeline.calc_driving_template(fps=fps, source_rgb_lst=[image], source_crop_info=source_crop_info, driving_rgb_lst=frames, driving_crop_info=driving_crop_info)
    print(f'{datetime.now()} >>>> calc driving template finished.')
    
    result = pipeline.animate(fps=fps, source_rgb_lst=[image], source_crop_info=source_crop_info, driving_template=driving_template)
    print(f'{datetime.now()} >>>> animate video finished.')
    #print(f'shape of result: {driving_template.shape}')

    frames = []
    images2video(images=result, wfp='../output_live.mp4', fps=fps)
    for i in track(range(total), description='Draw Landmarks....', transient=True):
        dst = driving_crop_info['frame_crop_lst'][i]
        lmk = driving_crop_info['lmk_crop_lst'][i]
        frame = draw_landmarks(frame=dst, landmarks=lmk)
        frames.append(frame)

    print(f'{datetime.now()} >>>> draw frames landmarks finished.')
    images2video(images=frames, wfp='../output_crop.mp4', fps=fps)


    
    #ffmpeg -i ../assets/dzq1.mp4 -i ../output_live.mp4 -c:v copy -c:a copy  ../output_audio.mp4