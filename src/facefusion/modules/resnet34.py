#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import onnxruntime
#from facefusion.utils import FaceMaskRegionMap, FaceMaskAllRegion
from ..utils.mask import FaceMaskRegionMap, FaceMaskAllRegion

class Resnet34:
    def __init__(self, model_path, providers):
        self.session  = onnxruntime.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        print(f'inputs[0].shape: {inputs[0].shape}')
        self.input_size = (inputs[0].shape[2], inputs[0].shape[3])
        self.input_name = inputs[0].name
        self.affine = False
    
    def pre_process(self, image):
        img = cv2.resize(image, self.input_size)
        img = img[:, :, ::-1].astype(np.float32) / 255
        img = np.subtract(img, np.array([ 0.485, 0.456, 0.406 ]).astype(np.float32))
        img = np.divide(img, np.array([ 0.229, 0.224, 0.225 ]).astype(np.float32))
        img = np.expand_dims(img, axis = 0)
        img = img.transpose(0, 3, 1, 2)
        return img
    
    def post_process(self, output, height, width):
        mask = cv2.resize(output.astype(np.float32), (width, height))
        mask = (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return mask
    
    def detect(self, image, regions):
        height, width = image.shape[0], image.shape[1]
        img = self.pre_process(image)
        #t = timeit.default_timer()
        outputs = self.session.run(None, {self.input_name: img})
        output = outputs[0][0]
        output = np.isin(output.argmax(0), [ FaceMaskRegionMap[region] for region in regions ])
        output = self.post_process(output, height, width)
        #print('infer time:',timeit.default_timer()-t)  
        return output


if __name__ == "__main__":
    
    from .yoloface import YoloFace
 
    def test_image(yolo, resnet):
        input_path = '../assets/liuyifei.jpg'
        #input_path = '/Users/wadahana/Desktop/mojing.jpg'
        output_path = '../output_mask.png'
        image = cv2.imread(input_path)
        face_list = yolo.detect(image=image, conf=0.7)
        face = face_list[0]
        x1, y1, x2, y2 = map(int, face[0])
        face_crop = image[y1:y2, x1:x2]
        resized_face = cv2.resize(face_crop, (256, 256))
        mask = resnet.detect(image=resized_face, regions=FaceMaskAllRegion)
        
        cv2.imwrite(output_path, (mask * 255).astype(np.uint8))
        
        
    providers = ['CPUExecutionProvider']
    yolo_path = '../../../models/facefusion/yoloface_8n.onnx'
    seg_path = '../../../models/facefusion/dfl_xseg.onnx'
    resnet_path = '../../../models/facefusion/bisenet_resnet_34.onnx'
    #model_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/dfl_xseg.onnx'
    
    yolo = YoloFace(model_path=yolo_path, providers=providers)
    resnet = Resnet34(model_path=resnet_path, providers=providers)

    test_image(yolo, resnet)