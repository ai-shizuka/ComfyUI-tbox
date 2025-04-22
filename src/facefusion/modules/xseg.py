#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import onnxruntime

class XSeg:
    def __init__(self, model_path, providers):
        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_profiling = True
        self.session  = onnxruntime.InferenceSession(model_path, sess_options, providers=providers)
        print(f'current providers: {self.session.get_providers()}') 
        print(f"available  providers: {onnxruntime.get_available_providers()}")
        
        for idx, node in enumerate(self.session.get_inputs()):
            print(f"[INPUT {idx}] Name: {node.name}, Type: {node.type}, Shape: {node.shape}")
    
        model_nodes = self.session.get_modelmeta()
        print("Model Meta:", model_nodes)
        
        inputs = self.session.get_inputs()
        self.input_size = (inputs[0].shape[1], inputs[0].shape[2])
        self.input_name = inputs[0].name
        self.affine = False
        
        # warmup
        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        inputs = {self.session.get_inputs()[0].name: self.pre_process(img_bgr)}
        output = self.session.run(None, inputs)
        profile_file = self.session.end_profiling()
        print(f"Profiling saved to: {profile_file}")
        


        # for node in self.session._sess.graph_view().nodes():
        #     print(f"{node.name}: {node.op_type}")
    
    def pre_process(self, image):
        img = cv2.resize(image, self.input_size)
        img = np.expand_dims(img, axis = 0).astype(np.float32) / 255
        img = img.transpose(0, 1, 2, 3)
        return img
    
    def post_process(self, output, height, width):
        mask = output.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        mask = cv2.resize(mask, (width, height))
        mask = (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return mask
    
    def detect(self, image):
        height, width = image.shape[0], image.shape[1]
        img = self.pre_process(image)
        #t = timeit.default_timer()
        outputs = self.session.run(None, {self.input_name: img})
        output = outputs[0][0]
        output = self.post_process(output, height, width)
        #print('infer time:',timeit.default_timer()-t)  
        return output
        
if __name__ == "__main__":
    from .yoloface import YoloFace
    from rich.progress import track
    from ..utils.affine import arcface_128_v2, ffhq_512, warp_face_by_landmark, paste_back, blend_frame
    
    import imageio
    import cv2
    import time
    import os
    
    os.environ["ORT_LOGGING_LEVEL"] = "VERBOSE"
        
    def overlay_mask_on_face(face_img, mask, alpha=0.5, color=(0, 0, 255)):
        overlay = face_img.copy()
        red_layer = np.full_like(face_img, color, dtype=np.uint8)
    
        # 扩展 mask 到 3 通道（方便融合）
        mask_3ch = np.stack([mask]*3, axis=-1).astype(np.uint8)

        # 只对 mask 区域应用 alpha 混合
        overlay = np.where(mask_3ch > 0, cv2.addWeighted(red_layer, alpha, face_img, 1 - alpha, 0), face_img)
        
        return overlay

    def test_image(yolo, xseg1, xseg2, input_path, output_path):
        image = cv2.imread(input_path)
        face_list = yolo.detect(image=image, conf=0.7)
        if face_list == None or len(face_list) == 0:
            print(f'no face detected in {input_path}')
            return
        face = face_list[0]
        x1, y1, x2, y2 = map(int, face[0])
        face_crop = image[y1:y2, x1:x2]
        resized_face = cv2.resize(face_crop, (256, 256))
        print(f'{time.time()}: xseg1 >>')
        mask1 = xseg1.detect(image=resized_face)
        print(f'{time.time()}: xseg1 <<')
        mask1 = (mask1 * 255).clip(0, 255).astype(np.uint8)
        print(f'{time.time()}: xseg2 >>')
        mask2 = xseg2.detect(image=resized_face)
        print(f'{time.time()}: xseg2 <<')
        mask2 = (mask2 * 255).clip(0, 255).astype(np.uint8)
        
        output1 = overlay_mask_on_face(resized_face, mask1, alpha=0.5, color=(0, 0, 255))
        output2 = overlay_mask_on_face(resized_face, mask2, alpha=0.5, color=(0, 0, 255))
#        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = cv2.hconcat([resized_face, output1, output2])
        print(f'output: {output_path}')
        cv2.imwrite(output_path, combined)
        
        
    def test_video(yolo, xseg1, xseg2, input_path, output_path):
        # input_path = '../suck2.mp4'
        # output_path = '../suck2_mask.mp4'
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
        writer = get_video_writer(output_path, fps)
        #while True:
        t = 0
        for i in track(range(total), description='Detecting....', transient=True):
            ret, frame = cap.read()
            if not ret:
                break
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face_list = yolo.detect(image=frame, conf=0.7)
            if face_list != None and len(face_list) > 0:
                start = time.time()
                face = face_list[0]
                resized_face, affine = warp_face_by_landmark(frame, face[1], arcface_128_v2, (256,256))
                
                x1, y1, x2, y2 = map(int, face[0])
                # face_crop = frame[y1:y2, x1:x2]
                # resized_face = cv2.resize(face_crop, (256, 256))
                mask1 = xseg1.detect(image=resized_face)
                mask2 = xseg2.detect(image=resized_face)
                #output = paste_back(frame, resized_face, mask, affine)
                
                stop = time.time()
                t = t + (stop - start)
                mask1 = (mask1 * 255).clip(0, 255).astype(np.uint8)
                #mask1 = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
                mask2 = (mask2 * 255).clip(0, 255).astype(np.uint8)
                #mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
                output1 = overlay_mask_on_face(resized_face, mask1, alpha=0.5, color=(0, 0, 255))
                output2 = overlay_mask_on_face(resized_face, mask2, alpha=0.5, color=(0, 0, 255))
                combined = cv2.hconcat([resized_face, output1, output2])
                
                #output = paste_back(frame, output, mask, affine)
                writer.append_data(combined[..., ::-1])
    
        cap.release()
        writer.close()
        print(f'total time: {t:.4f} sec; total frames: {total}; average time per frame: {t/total:.4f} sec')
        
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
        
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']

    # providers = [
    #     ('TensorrtExecutionProvider', {
    #         'trt_engine_cache_enable': True,
    #         'trt_engine_cache_path': './trt_cache'
    #     }),
    #     'CUDAExecutionProvider',  # 备份
    #     'CPUExecutionProvider'
    # ]
    
    yolo_path = '../../../models/facefusion/yoloface_8n.onnx'
    seg_path = '../../../models/facefusion/dfl_xseg.onnx'
    seg1_path = '../../../models/facefusion/xseg_1_sim.onnx'
    #seg1_path = '../../../models/facefusion/xseg_1.onnx'
    
    yolo = YoloFace(model_path=yolo_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f'{time.time()}: xseg 1 start >>')
    xseg = XSeg(model_path=seg_path, providers=providers)
    print(f'{time.time()}: xseg 1 end >>')
    
    print(f'{time.time()}: xseg 2 start >>')
    xseg1 = XSeg(model_path=seg1_path, providers=providers)
    print(f'{time.time()}: xseg 2 end >>')
    
    # p = '/home/eric/workspace/AI/sd/temp/mask/0d9c272ae90f50f1795a2992a9483fcc/output_mask.mp4'
    # v = '/home/eric/workspace/AI/sd/temp/mask/0d9c272ae90f50f1795a2992a9483fcc/target.mp4'
    # test_video(yolo, xseg, xseg1, v,  p)
    
    root_path = "/home/eric/workspace/AI/sd/temp/mask"
    image_list = []
    video_list = []
    for subdir in os.listdir(root_path):
        subpath = os.path.join(root_path, subdir)
        if os.path.isdir(subpath) == False :
            continue    
        target_path = os.path.join(subpath, 'target.mp4')
        if os.path.isfile(target_path):
            video_list.append(subpath)
            continue
        target_path = os.path.join(subpath, 'target.jpg')
        if os.path.isfile(target_path):
            image_list.append(subpath)
            
    print(f'image_list:')         
    for subpath in image_list:
        input_path = os.path.join(subpath, 'target.jpg')
        print(f'input_path: {input_path}')
        test_image(yolo, xseg, xseg1, input_path, os.path.join(subpath, 'output_mask.png'))
    
    print(f'video_list:')    
    for subpath in video_list:
        input_path = os.path.join(subpath, 'target.mp4')
        print(f'input_path: {input_path}')
        #test_video(yolo, xseg, xseg1, input_path,  os.path.join(subpath, 'output_mask.mp4'))
            
    # input_path = '/home/eric/workspace/AI/sd/temp/mask/0ef45196ed648cb592f89dd89d436dec/target.jpg'
    # output_path = '/home/eric/workspace/AI/sd/temp/mask/0ef45196ed648cb592f89dd89d436dec/'
    # test_image(yolo, xseg, input_path, os.path.join(output_path, 'output_mask_0.png'))
    # test_image(yolo, xseg1, input_path, os.path.join(output_path, 'output_mask_1.png'))
    

    


    
    #providers=['CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider']
   
    