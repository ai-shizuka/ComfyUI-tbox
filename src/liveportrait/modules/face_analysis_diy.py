# coding: utf-8

"""
face detectoin and alignment using InsightFace
"""

import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from liveportrait.utils.timer import Timer
from liveportrait.utils.io import contiguous

def sort_by_direction(faces, direction: str = 'large-small', face_center=None):
    if len(faces) <= 0:
        return faces

    if direction == 'left-right':
        return sorted(faces, key=lambda face: face['bbox'][0])
    if direction == 'right-left':
        return sorted(faces, key=lambda face: face['bbox'][0], reverse=True)
    if direction == 'top-bottom':
        return sorted(faces, key=lambda face: face['bbox'][1])
    if direction == 'bottom-top':
        return sorted(faces, key=lambda face: face['bbox'][1], reverse=True)
    if direction == 'small-large':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
    if direction == 'large-small':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]), reverse=True)
    if direction == 'distance-from-retarget-face':
        return sorted(faces, key=lambda face: (((face['bbox'][2]+face['bbox'][0])/2-face_center[0])**2+((face['bbox'][3]+face['bbox'][1])/2-face_center[1])**2)**0.5)
    return faces


class FaceAnalysisDIY(FaceAnalysis):
    def __init__(self, name='buffalo_l', root='~/.insightface', allowed_modules=None, **kwargs):
        super().__init__(name=name, root=root, allowed_modules=allowed_modules, **kwargs)

        self.timer = Timer()

    def get(self, img_bgr, **kwargs):
        max_num = kwargs.get('max_face_num', 0)  # the number of the detected faces, 0 means no limit
        flag_do_landmark_2d_106 = kwargs.get('flag_do_landmark_2d_106', True)  # whether to do 106-point detection
        direction = kwargs.get('direction', 'large-small')  # sorting direction
        face_center = None

        bboxes, kpss = self.det_model.detect(img_bgr, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue

                if (not flag_do_landmark_2d_106) and taskname == 'landmark_2d_106':
                    continue

                # print(f'taskname: {taskname}')
                model.get(img_bgr, face)
            ret.append(face)

        ret = sort_by_direction(ret, direction, face_center)
        return ret

    def warmup(self):
        self.timer.tic()

        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        self.get(img_bgr)

        elapse = self.timer.toc()
        print(f'FaceAnalysisDIY warmup time: {elapse:.3f}s')




if __name__ == '__main__':
    import cv2
    from rich.progress import track
    from liveportrait.utils.helper import draw_landmarks
    from liveportrait.utils.video import images2video
    
    def test_image(detecter) :
        input_path = '../assets/liuyifei.jpg'
        image = cv2.imread(input_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_list = detecter.get(img_bgr=contiguous(image[..., ::-1]))
        face = face_list[0]
        #print(f'face: {face}')
        frame = draw_landmarks(frame=image, landmarks=face.landmark_2d_106)
        x1, y1, x2, y2 = map(int, face.bbox) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        cv2.imwrite(f'/Users/wadahana/Desktop/output_crop.jpg', frame)
    

        
    def test_video(detecter) :
        video_input = '/Users/wadahana/Desktop/sis/faceswap/test/sq/suck2/suck2-short.mp4'
        
        cap = cv2.VideoCapture(video_input)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        frames = []
        #while True:
        for i in track(range(total), description='Detecting....', transient=True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face_list = detecter.get(img_bgr=contiguous(frame[..., ::-1]))
            if len(face_list) == 0:
                continue
            face = face_list[0]
            frame = draw_landmarks(frame, face.landmark_2d_106)
            x1, y1, x2, y2 = map(int, face.bbox) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
            frames.append(frame)
            
        cap.release()
        images2video(images=frames, wfp='../output_insightface.mp4', fps=fps)
   
    model_path = '../../../models/liveportrait/landmark.onnx'
    providers = [ "CPUExecutionProvider"] #"CoreMLExecutionProvider",
    detecter = FaceAnalysisDIY( name="buffalo_l",
                    root='../../../models/insightface',
                    providers=providers)
    detecter.prepare(ctx_id=0, det_size=(512, 512), det_thresh=0.1)
    detecter.warmup()
    
    test_video(detecter)
    