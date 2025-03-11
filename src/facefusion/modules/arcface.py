#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import onnxruntime
from facefusion.utils.affine import arcface_112_v2, warp_face_by_landmark

class ArcFaceW600k:
    def __init__(self, model_path, providers):
        self.session  = onnxruntime.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        self.input_size = (inputs[0].shape[2], inputs[0].shape[3])
        self.input_name = inputs[0].name
        self.affine = False
    
    def pre_process(self, image):
        image = image / 127.5 - 1
        image = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        image = np.expand_dims(image, axis = 0)
        return image
    
    def post_process(self, result):
        embedding = result[0].ravel()
        normed_embedding = embedding / np.linalg.norm(embedding)
        return (embedding, normed_embedding)
    
    def embedding(self, image, landmarks):
        image, _ = warp_face_by_landmark(image, landmarks, arcface_112_v2, self.input_size)
        img = self.pre_process(image)
        #t = timeit.default_timer()
        result = self.session.run(None, {self.input_name: img})
        return self.post_process(result)
    
def face_distance(e0, e1):
    #print(f'e0: {e0}')
    return 1 - np.dot(e0[2], e1[2])

if __name__ == "__main__":
    from yoloface import YoloFace
    import itertools
    
    def calc_face_embedding(image, yolo, recognizer):
        face_list = yolo.detect(image=image, conf=0.7)
        face = face_list[0]
        embedding = recognizer.embedding(image=image, landmarks=face[1])
        return embedding
    
    def test_image(yolo, recognizer):
        embedding_list = []
        for i in range(0, 5):
            input_path = f'../assets/risa_{i}.jpg'
            image = cv2.imread(input_path)
            e = calc_face_embedding(image, yolo, recognizer)
            x = (f'risa_{i}', e[0], e[1])
            embedding_list.append(x)
        for i in range(0, 5):
            input_path = f'../assets/lor_{i}.jpg'
            image = cv2.imread(input_path)
            e = calc_face_embedding(image, yolo, recognizer)
            x = (f'lor_{i}', e[0], e[1])
            embedding_list.append(x)
            
        # 计算所有两两组合的距离
        distances = []
        for e0, e1 in itertools.combinations(embedding_list, 2):
            dist = face_distance(e0, e1)
            distances.append((e0, e1, dist))
        # 打印距离
        for e0, e1, dist in distances:
            print(f"Distance between {e0[0]} and {e1[0]}: {dist}")
        
    providers = ['CPUExecutionProvider']
    yolo_path = '../../../models/facefusion/yoloface_8n.onnx'
    seg_path = '../../../models/facefusion/dfl_xseg.onnx'
    arcface_w600k_path = '../../../models/facefusion/arcface_w600k_r50.onnx'
    #model_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/dfl_xseg.onnx'
    
    yolo = YoloFace(model_path=yolo_path, providers=providers)
    arcface = ArcFaceW600k(model_path=arcface_w600k_path, providers=providers)

    test_image(yolo, arcface)
