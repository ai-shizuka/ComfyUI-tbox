import os
import onnx
import onnxruntime
import cv2
import numpy as np
from collections import namedtuple

Face = namedtuple('Face',
[
	'bounding_box',
	'landmarks',
	'scores',
])
 
class YoloFaceOnnx:
    def __init__(self, model_path, providers):
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        self.input_size = (inputs[0].shape[2], inputs[0].shape[3])
        self.input_name = inputs[0].name
        
    def pre_process(self, image):
        img = cv2.resize(image, self.input_size)
        img = (img - 127.5) / 128.0
        img = np.expand_dims(img.transpose(2, 0, 1), axis = 0).astype(np.float32)
        return img
        
    def post_process(self, size, bounding_box_list, face_landmark_5_list, score_list):
  
        sort_indices = np.argsort(-np.array(score_list))
        bounding_box_list = [ bounding_box_list[index] for index in sort_indices ]
        face_landmark_5_list = [face_landmark_5_list[index] for index in sort_indices]
        score_list = [ score_list[index] for index in sort_indices ]
        
        face_list = []
        keep_indices = self.apply_nms(bounding_box_list, 0.4)
        for index in keep_indices:
            bounding_box = bounding_box_list[index]
            face_landmark = face_landmark_5_list[index]
            score = score_list[index],
            #print(f'bounding_box  >> : {bounding_box}')
            face_list.append(Face(
				bounding_box = self.expand_bounding_box(size, bounding_box),
				landmarks = face_landmark,
				scores = score,
			))
        return face_list
            
    def apply_nms(self, bounding_box_list, iou_threshold):
        keep_indices = []
        dimension_list = np.reshape(bounding_box_list, (-1, 4))
        x1 = dimension_list[:, 0]
        y1 = dimension_list[:, 1]
        x2 = dimension_list[:, 2]
        y2 = dimension_list[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.arange(len(bounding_box_list))
        while indices.size > 0:
            index = indices[0]
            remain_indices = indices[1:]
            keep_indices.append(index)
            xx1 = np.maximum(x1[index], x1[remain_indices])
            yy1 = np.maximum(y1[index], y1[remain_indices])
            xx2 = np.minimum(x2[index], x2[remain_indices])
            yy2 = np.minimum(y2[index], y2[remain_indices])
            width = np.maximum(0, xx2 - xx1 + 1)
            height = np.maximum(0, yy2 - yy1 + 1)
            iou = width * height / (areas[index] + areas[remain_indices] - width * height)
            indices = indices[np.where(iou <= iou_threshold)[0] + 1]
        return keep_indices
    
    def expand_bounding_box(self, size, bbox, expansion=32):
        """
        根据给定的 bbox，在四个边各扩展一定像素。
        
        :param image: 输入的图像
        :param bbox: 原始边界框，格式为 (x1, y1, x2, y2)
        :param expansion: 每个边扩展的像素数
        :return: 扩展后的图像
        """
        x1, y1, x2, y2 = bbox
        # 扩展边界框
        x1_expanded = max(x1 - expansion, 0)  # 保证不越界
        y1_expanded = max(y1 - expansion, 0)
        x2_expanded = min(x2 + expansion, size[1])  # 保证不越界
        y2_expanded = min(y2 + expansion, size[0])
        return [x1_expanded, y1_expanded, x2_expanded, y2_expanded]

    def detect(self, image, conf):
        img = self.pre_process(image)
        ratio_height = image.shape[0] / img.shape[2]
        ratio_width = image.shape[1] / img.shape[3]
        outputs = self.session.run(None, {self.input_name: img})
        outputs = np.squeeze(outputs).T
        bounding_box_raw, score_raw, face_landmark_5_raw = np.split(outputs, [ 4, 5 ], axis = 1)
        
        bounding_box_list = []
        face_landmark_5_list = []
        score_list = []
        keep_indices = np.where(score_raw > conf)[0]
        if keep_indices.any():
            bounding_box_raw, face_landmark_5_raw, score_raw = bounding_box_raw[keep_indices], face_landmark_5_raw[keep_indices], score_raw[keep_indices]
            for bounding_box in bounding_box_raw:
                bounding_box_list.append(np.array(
                [
                    (bounding_box[0] - bounding_box[2] / 2) * ratio_width,
                    (bounding_box[1] - bounding_box[3] / 2) * ratio_height,
                    (bounding_box[0] + bounding_box[2] / 2) * ratio_width,
                    (bounding_box[1] + bounding_box[3] / 2) * ratio_height
                ]))
            face_landmark_5_raw[:, 0::3] = (face_landmark_5_raw[:, 0::3]) * ratio_width
            face_landmark_5_raw[:, 1::3] = (face_landmark_5_raw[:, 1::3]) * ratio_height
            for face_landmark_5 in face_landmark_5_raw:
                face_landmark_5_list.append(np.array(face_landmark_5.reshape(-1, 3)[:, :2]))
            score_list = score_raw.ravel().tolist()
        
        return self.post_process(image.shape, bounding_box_list, face_landmark_5_list, score_list)
    

if __name__ == "__main__":
    model_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'
    providers=['CPUExecutionProvider']
   
    detector = YoloFaceOnnx(model_path=model_path, providers=providers)

    image = cv2.imread('/Users/wadahana/Desktop/test4.jpg')

    face_list = detector.detect(image=image, conf=0.7)
    face = face_list[0]
    
    pt1 = (int(face.bounding_box[0]), int(face.bounding_box[1]))  # 左上角 (x1, y1)
    pt2 = (int(face.bounding_box[2]), int(face.bounding_box[3]))  # 右下角 (x2, y2)
    cv2.rectangle(image, pt1, pt2, (255, 0, 0), 1)

    print(f'face_list: {face_list}')
    cv2.imwrite('/Users/wadahana/Desktop/output.jpg', image)

    