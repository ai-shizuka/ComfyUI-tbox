import os
import cv2
import onnx
import onnxruntime
import numpy as np
from collections import namedtuple
from typing import List, Literal

Face = namedtuple('Face',
[
	'bounding_box',
	'landmarks',
	'scores',
])
 
FaceAnalyserOrder = Literal['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small', 'best-worst', 'worst-best']

class YoloFace:
    def __init__(self, model_path, providers):
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        self.input_size = (inputs[0].shape[2], inputs[0].shape[3])
        self.input_name = inputs[0].name
        
    def resize_frame_resolution(self, vision_frame , max_resolution):
        height, width = vision_frame.shape[:2]
        max_width, max_height = max_resolution

        if height > max_height or width > max_width:
            scale = min(max_height / height, max_width / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(vision_frame, (new_width, new_height))
        return vision_frame
    
    def pre_process(self, image):
        img = np.zeros((self.input_size[0], self.input_size[1], 3))
        img[:image.shape[0], :image.shape[1], :] = image
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
            face_list.append((
				self.expand_bounding_box(size, bounding_box),
				face_landmark,
				score,
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

    def detect(self, image, conf, order='left-right'):
        img = self.resize_frame_resolution(image, self.input_size)
        detect_img = self.pre_process(img)
        ratio_height = image.shape[0] / img.shape[0]
        ratio_width = image.shape[1] / img.shape[1]
        
        outputs = self.session.run(None, {self.input_name: detect_img})
        
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
        
        faces = self.post_process(image.shape, bounding_box_list, face_landmark_5_list, score_list)
        if len(faces) > 1:
            faces = self.sort_by_order(faces, order)
        return faces 
    
    def sort_by_order(self, faces : List[Face], order : FaceAnalyserOrder) -> List[Face]:
        if order == 'left-right':
            return sorted(faces, key = lambda face: face[0][0])
        if order == 'right-left':
            return sorted(faces, key = lambda face: face[0][0], reverse = True)
        if order == 'top-bottom':
            return sorted(faces, key = lambda face: face[0][1])
        if order == 'bottom-top':
            return sorted(faces, key = lambda face: face[0][1], reverse = True)
        if order == 'small-large':
            return sorted(faces, key = lambda face: (face[0][2] - face[0][0]) * (face[0][3] - face[0][1]))
        if order == 'large-small':
            return sorted(faces, key = lambda face: (face[0][2] - face[0][0]) * (face[0][3] - face[0][1]), reverse = True)
        if order == 'best-worst':
            return sorted(faces, key = lambda face: face[2], reverse = True)
        if order == 'worst-best':
            return sorted(faces, key = lambda face: face[2])
        return faces

if __name__ == "__main__":
    from liveportrait.utils.helper import draw_landmarks
    from liveportrait.utils.video import images2video
    from rich.progress import track
    
    def test_image(detector):
        #input = '/Users/wadahana/Desktop/ad_enhance-d77b92ad.png'
        input = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250405/ec6ee635b4742b08e0fdea6c03769514/source.jpg"
        image = cv2.imread(input)
        face_list = detector.detect(image=image, conf=0.5, order='best-worst')
        face = face_list[0]
        res = [512, 512]
        #box_mask = create_box_mask(res, 0.3, (0,0,0,0))
        #crop_mask = np.minimum.reduce([box_mask]).clip(0, 1)
        image = draw_landmarks(image, face[1])

        x1, y1, x2, y2 = map(int, face[0])
        face_crop = image[y1:y2, x1:x2]
        cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 1)
        resized_face = cv2.resize(face_crop, (512, 512))
        cv2.imwrite('/Users/wadahana/Desktop/output.jpg', image)
        #cv2.imwrite('/Users/wadahana/Desktop/output_mask_png', crop_mask)
        
    def adjust_bounding_box(bbox, width, height, dsize=512):
        x1, y1, x2, y2 = map(int, bbox)

        # 计算中心点
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 计算新的边界框
        new_x1 = max(0, cx - dsize // 2)
        new_y1 = max(0, cy - dsize // 2)
        new_x2 = min(width, cx + dsize // 2)
        new_y2 = min(height, cy + dsize // 2)

        # 如果因超出边界导致尺寸不足512x512，进行调整
        if new_x2 - new_x1 < dsize:
            if new_x1 == 0:
                new_x2 = min(width, new_x1 + dsize)
            else:
                new_x1 = max(0, new_x2 - dsize)

        if new_y2 - new_y1 < dsize:
            if new_y1 == 0:
                new_y2 = min(height, new_y1 + dsize)
            else:
                new_y1 = max(0, new_y2 - dsize)

        return (new_x1, new_y1, new_x2, new_y2)

    def test_video(detector):
        from facefusion.utils.affine import warp_face_by_landmark, paste_back
        #video_input = '../assets/dzq.mp4'
        
        video_input = '/Users/wadahana/Desktop/sis/faceswap/test/sq/suck2/suck2-short.mp4'
        cap = cv2.VideoCapture(video_input)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
        frames = []
        #while True:
        for i in track(range(total), description='Detecting....', transient=True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face_list = detector.detect(image=frame, conf=0.7)
            if len(face_list) == 0:
                continue
            face = face_list[0]
            frame = draw_landmarks(frame, face[1])
            x1, y1, x2, y2 = map(int, face[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 

            #(x1, y1, x2, y2) = adjust_bounding_box(bbox=face.bounding_box, width=width, height=height, dsize=512)
            face_crop = frame[y1:y2, x1:x2]
            resized_face = cv2.resize(face_crop, (512, 512))
            #frames.append(resized_face)
            #out.write(resized_face)
            
            frames.append(frame)
    
        images2video(frames, wfp='../output_yoloface.mp4', fps=fps)
        cap.release()



    model_path = '../../../models/facefusion/yoloface_8n.onnx'
    providers=['CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider']
   
    detector = YoloFace(model_path=model_path, providers=providers)
    test_image(detector)
    print('test yoloface_onnx finished! ')
