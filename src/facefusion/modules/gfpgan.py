#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import argparse
import imageio
import numpy as np
import onnxruntime

class GFPGAN:
    def __init__(self, model_path, providers):
        self.session  = onnxruntime.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        for input in inputs:
            if input.name == 'input':
                self.input_size = (inputs[0].shape[2], inputs[0].shape[3])
        self.affine = False
    
    def pre_process(self, image):
        image = cv2.resize(image, self.input_size)
        image = image[:, :, ::-1] / 255.0
        image = (image - 0.5) / 0.5
        image = np.expand_dims(image.transpose(2, 0, 1), axis = 0).astype(np.float32)
        return image

    def post_process(self, output, height, width):
        output = output.clip(-1,1)
        output = (output + 1) / 2
        output = output.transpose(1, 2, 0)
        output = (output * 255.0).round()
        output = output.astype(np.uint8)[:, :, ::-1]
        output = cv2.resize(output, (width, height))
        return output

    def run(self, image):
        height, width = image.shape[0], image.shape[1]
        img = self.pre_process(image)
        #t = timeit.default_timer()
        outputs = self.session.run(None, {'input': img})
        output = outputs[0][0]
        output = self.post_process(output, height, width)
        #print('infer time:',timeit.default_timer()-t)  
        output = output.astype(np.uint8)
        return output

if __name__ == "__main__":
    from .yoloface import YoloFace
    from facefusion.utils.affine import ffhq_512, warp_face_by_landmark, paste_back, blend_frame
    from facefusion.utils.mask import create_bbox_mask
    
    def restore_face(yolo, gfpgan, image):
        face_list = yolo.detect(image=image, conf=0.7)
        #print(f'restore_face >> total of face: {len(face_list)}')
        for index, face in enumerate(face_list):
            cropped, affine_matrix = warp_face_by_landmark(image, face[1], ffhq_512, gfpgan.input_size)
            box_mask = create_bbox_mask(gfpgan.input_size, 0.3, (0,0,0,0))
            crop_mask = np.minimum.reduce([box_mask]).clip(0, 1)
            result = gfpgan.run(cropped)
            
            pasted = paste_back(image, result, crop_mask, affine_matrix)
            output = blend_frame(image, pasted, 0.8)
            
        return output

    def get_video_writer(outout_path, fps):
        video_format = 'mp4'     # default is mp4 format
        codec = 'libx264'        # default is libx264 encoding
        #quality = quality        # video quality
        pixelformat = 'yuv420p'  # video pixel format
        image_mode = 'rbg'
        macro_block_size = 2
        ffmpeg_params = ['-crf', '20']
        writer = imageio.get_writer(uri=outout_path,
                            format=video_format,
                            fps=fps, 
                            codec=codec, 
                            #quality=quality, 
                            ffmpeg_params=ffmpeg_params, 
                            pixelformat=pixelformat, 
                            macro_block_size=macro_block_size)
        return writer
    
    def test_image(yolo, gfpgan, input_path, output_path):
        image = cv2.imread(input_path)
        output = restore_face(yolo, gfpgan, image)
        cv2.imwrite(output_path, output)
        
    def test_video(yolo, gfpgan, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
        target_fps = min(50, fps)
        frame_interval = fps / target_fps 
        frame_index = 0
        new_frame_id = 0  # 目标视频的帧编号
        writer = get_video_writer(output_path, target_fps)
        
        while cap.isOpened():
            ret, target = cap.read()
            if not ret:
                break
            if new_frame_id * frame_interval <= frame_index:
                frame = target
                output = restore_face(yolo=yolo, gfpgan=gfpgan, image=frame)
                #writer.append_data(output) 
                writer.append_data(output[..., ::-1])
                new_frame_id += 1

            frame_index += 1
            
        writer.close()
        cap.release()
        
    providers  = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    model_path = '../../../models/facefusion/gfpgan_1.4.onnx'
    yolo_path  = '../../../models/facefusion/yoloface_8n.onnx'
    
    yolo = YoloFace(model_path=yolo_path, providers=providers)
    gfpgan = GFPGAN(model_path=model_path, providers=providers)

    input_image = '/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250408/20edc6a339e5c892a5f6f9021ee7b435/target.jpg'
    input_video = '/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250408/0eeb9e938dbfaf1a5914ef5d6ef27496/target.mp4'
    
    output_image = '/Users/wadahana/Desktop/output.jpg'
    output_video = '/Users/wadahana/Desktop/output.mp4'
    
    test_video(yolo, gfpgan, input_video, output_video)
 