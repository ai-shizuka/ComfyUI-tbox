import cv2
import numpy as np

ffhq_512 = np.array([
    [ 0.37691676, 0.46864664 ],
    [ 0.62285697, 0.46912813 ],
    [ 0.50123859, 0.61331904 ],
    [ 0.39308822, 0.72541100 ],
    [ 0.61150205, 0.72490465 ]
])
arcface_112_v2 = np.array([
    [ 0.34191607, 0.46157411 ],
    [ 0.65653393, 0.45983393 ],
    [ 0.50022500, 0.64050536 ],
    [ 0.37097589, 0.82469196 ],
    [ 0.63151696, 0.82325089 ]
])
arcface_128_v2 = np.array([
    [ 0.36167656, 0.40387734 ],
    [ 0.63696719, 0.40235469 ],
    [ 0.50019687, 0.56044219 ],
    [ 0.38710391, 0.72160547 ],
    [ 0.61507734, 0.72034453 ]
])

def warp_face_by_landmark(image , face_landmark_5, warp_template, crop_size):
	normed_warp_template = warp_template * crop_size
	affine_matrix = cv2.estimateAffinePartial2D(face_landmark_5, normed_warp_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
	cropped = cv2.warpAffine(image, affine_matrix, crop_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
	return cropped, affine_matrix


def paste_back(image, cropped, crop_mask, affine_matrix):
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_size = image.shape[:2][::-1]
    inverse_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_size).clip(0, 1)
    inverse_vision_frame = cv2.warpAffine(cropped, inverse_matrix, temp_size, borderMode = cv2.BORDER_REPLICATE)
    paste_vision_frame = image.copy()
    paste_vision_frame[:, :, 0] = inverse_mask * inverse_vision_frame[:, :, 0] + (1 - inverse_mask) * image[:, :, 0]
    paste_vision_frame[:, :, 1] = inverse_mask * inverse_vision_frame[:, :, 1] + (1 - inverse_mask) * image[:, :, 1]
    paste_vision_frame[:, :, 2] = inverse_mask * inverse_vision_frame[:, :, 2] + (1 - inverse_mask) * image[:, :, 2]
    return paste_vision_frame

def blend_frame(origin_image , new_image, blend ):
	face_enhancer_blend = 1 - (blend)
	temp_vision_frame = cv2.addWeighted(origin_image, face_enhancer_blend, new_image, blend, 0)
	return temp_vision_frame