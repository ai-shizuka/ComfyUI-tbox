#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import onnxruntime

class GFPGAN:
    def __init__(self, model_path, providers):
        self.session  = onnxruntime.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        for input in inputs:
            print(f'GFPGAN >> input: {input.name}, shape: {input.shape}')
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
    
    providers=['CPUExecutionProvider']
    model_path = '../../../models/facefusion/gfpgan_1.4.onnx'
    yolo_path = '../../../models/facefusion/yoloface_8n.onnx'
    
    detector = YoloFace(model_path=yolo_path, providers=providers)
    session = GFPGAN(model_path=model_path, providers=providers)

    image = cv2.imread('/Users/wadahana/Desktop/oo1.png')

    face_list = detector.detect(image=image, conf=0.7)
    print(f'total of face: {len(face_list)}')
    
    output = image
    for index, face in enumerate(face_list):
        cropped, affine_matrix = warp_face_by_landmark(image, face[1], ffhq_512, session.input_size)
        box_mask = create_bbox_mask(session.input_size, 0.3, (0,0,0,0))
        crop_mask = np.minimum.reduce([box_mask]).clip(0, 1)
        result = session.run(cropped)
        cv2.imwrite(f'/Users/wadahana/Desktop/output_{index}.jpg', result)
        pasted = paste_back(output, result, crop_mask, affine_matrix)
        output = blend_frame(output, pasted, 0.8)
        
    cv2.imwrite(f'/Users/wadahana/Desktop/output.jpg', output)
 