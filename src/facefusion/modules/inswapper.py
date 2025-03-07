#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import onnx
import onnxruntime
from functools import lru_cache

def split_into_tiles(image, total_tiles, model_size):
    image = image.reshape(model_size[0], total_tiles, model_size[1], total_tiles, 3)
    image = image.transpose(1, 3, 0, 2, 4).reshape(total_tiles ** 2, model_size[0], model_size[1], 3)
    return image

def merge_from_tiles(image, total_tiles, model_size, dsize):
    image = np.stack(image, axis = 0).reshape(total_tiles, total_tiles, model_size[0], model_size[1], 3)
    image = image.transpose(2, 0, 3, 1, 4).reshape(dsize[0], dsize[1], 3)
    return image

class InSwapper:
    def __init__(self, model_path, providers):
        self.mean = [ 0.0, 0.0, 0.0 ],
        self.standard_deviation = [ 1.0, 1.0, 1.0 ]
        self.model_path = model_path
        self.session  = onnxruntime.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        for input in inputs:
            print(f'input: {input.name}, shape: {input.shape}')
            if input.name == 'target':
                self.target_size = (input.shape[2], input.shape[3])
            elif input.name == 'source':
                self.source_size = input.shape[1]
        self.affine = False
    
    @lru_cache(maxsize = None)
    def get_model_initializer(self):
        model = onnx.load(self.model_path)
        return onnx.numpy_helper.to_array(model.graph.initializer[-1])


    def pre_process_source(self, embedding):
        model_initializer = self.get_model_initializer()
        source_embedding = embedding.reshape((1, -1))
        source_embedding = np.dot(source_embedding, model_initializer) / np.linalg.norm(source_embedding)
        return source_embedding
    
    def pre_process_target(self, image):
        image = image[:, :, ::-1] / 255.0
        image = (image - self.mean) / self.standard_deviation
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis = 0).astype(np.float32)
        return image
    
    def post_process(self, output):
        output = output.transpose(1, 2, 0)
        output = output.clip(0, 1)
        output = output[:, :, ::-1] * 255
        return output
    
    def swap(self, source, target):
        height, width = target.shape[0], target.shape[1]
        total_tiles = height // self.target_size[0]
        source = self.pre_process_source(source)
        target_frames = split_into_tiles(target, total_tiles=total_tiles, model_size=self.target_size)
        results = []
        print(f'target_frames.shape: {target_frames.shape}')
        for target_frame in target_frames:
            inputs = {}
            inputs['target'] = self.pre_process_target(target_frame)
            inputs['source'] = source 
            result = self.session.run(None, inputs)
            output = self.post_process(result[0][0])
            print(f'output.shape: {target_frames.shape}')
            results.append(output)
#        print(f'results.shape: {results.shape}')
        output = merge_from_tiles(results, total_tiles=total_tiles, model_size=self.target_size, dsize=target.shape)
        return output


if __name__ == "__main__":
#    from liveportrait.utils.landmark_runner import draw_landmarks
    from liveportrait.utils.video import images2video
    from facefusion.affine import arcface_128_v2, warp_face_by_landmark, create_box_mask, paste_back, blend_frame
    from rich.progress import track
    from .yoloface import YoloFace
    from .arcface import ArcFaceW600k
    import itertools
    
    def calc_face_embedding(image, yolo, recognizer):
        face_list = yolo.detect(image=image, conf=0.7)
        face = face_list[0]
        embedding = recognizer.embedding(image=image, landmarks=face[1])
        return embedding
    
    def crop_face(image, yolo, dsize):
        face_list = yolo.detect(image=image, conf=0.7)
        face = face_list[0]
        cropped, affine_matrix = warp_face_by_landmark(image, face[1], arcface_128_v2, dsize)
        
        return cropped, affine_matrix
    
    def test_image(yolo, recognizer, inswapper):
        source_input_path = f'../assets/risa_1.jpg'
        target_input_path = f'../assets/lor_1.jpg'
        
        source = cv2.imread(source_input_path)
        source_embedding = calc_face_embedding(source, yolo, recognizer)
        
        target = cv2.imread(target_input_path)
        face, affine = crop_face(target, yolo, (256, 256))
        output = inswapper.swap(source_embedding[0], face)
        
        cv2.imwrite('../output_swapi.jpg', output)
        
        box_mask = create_box_mask((256, 256), 0.3, (0,0,0,0))
        crop_mask = np.minimum.reduce([box_mask]).clip(0, 1)
        output = paste_back(target, output, crop_mask, affine)
        cv2.imwrite('../output_swapi2.jpg', output)
        
    providers = ['CPUExecutionProvider']
    yolo_path = '../../../models/facefusion/yoloface_8n.onnx'
    inswapper_path = '../../../models/facefusion/inswapper_128.onnx'
    arcface_w600k_path = '../../../models/facefusion/arcface_w600k_r50.onnx'
    
    #model_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/dfl_xseg.onnx'
    
    yolo = YoloFace(model_path=yolo_path, providers=providers)
    inswapper = InSwapper(model_path=inswapper_path, providers=providers)
    arcface = ArcFaceW600k(model_path=arcface_w600k_path, providers=providers)
    test_image(yolo, arcface, inswapper)