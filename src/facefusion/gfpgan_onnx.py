#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import timeit
import onnxruntime
from facefusion.affine import create_box_mask, warp_face_by_landmark, paste_back

class GFPGANOnnx:
    def __init__(self, model_path, providers):
        self.session  = onnxruntime.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        self.input_size = (inputs[0].shape[2], inputs[0].shape[3])
        self.input_name = inputs[0].name
        self.affine = False
    
    def pre_process(self, image):
        img = cv2.resize(image, self.input_size)
        img = img/255.0
        img[:,:,0] = (img[:,:,0]-0.5)/0.5
        img[:,:,1] = (img[:,:,1]-0.5)/0.5
        img[:,:,2] = (img[:,:,2]-0.5)/0.5
        img = np.float32(img[np.newaxis,:,:,:])
        img = img.transpose(0, 3, 1, 2)
        return img

    def post_process(self, output, height, width):
        output = output.clip(-1,1)
        output = (output + 1) / 2
        output = output.transpose(1, 2, 0)
       # output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = (output * 255.0).round()
        output = cv2.resize(output, (width, height))
        return output

    def run(self, image):
        height, width = image.shape[0], image.shape[1]
        img = self.pre_process(image)
        t = timeit.default_timer()
        outputs = self.session.run(None, {self.input_name: img})
        output = outputs[0][0]
        output = self.post_process(output, height, width)
        print('infer time:',timeit.default_timer()-t)  
        output = output.astype(np.uint8)
        return output

if __name__ == "__main__":
    from yoloface_onnx import YoloFaceOnnx
    providers=['CPUExecutionProvider']
    model_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/gfpgan_1.4.onnx'
    yolo_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'
    
    detector = YoloFaceOnnx(model_path=yolo_path, providers=providers)
    session = GFPGANOnnx(model_path=model_path, providers=providers)

   

    image = cv2.imread('/Users/wadahana/Desktop/anime-3.jpeg')

    face_list = detector.detect(image=image, conf=0.7)
    print(f'total of face: {len(face_list)}')
    
    output = image
    for index, face in enumerate(face_list):
        cropped, affine_matrix = warp_face_by_landmark(image, face.landmarks, session.input_size)
        box_mask = create_box_mask(session.input_size, 0.3, (0,0,0,0))
        crop_mask = np.minimum.reduce([box_mask]).clip(0, 1)
        result = session.run(cropped)
        cv2.imwrite(f'/Users/wadahana/Desktop/output_{index}.jpg', result)
        output = paste_back(output, result, crop_mask, affine_matrix)
        
    cv2.imwrite(f'/Users/wadahana/Desktop/output.jpg', output)
 