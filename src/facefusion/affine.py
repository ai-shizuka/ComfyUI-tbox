import cv2
import numpy as np

ffhq_512 = np.array([
    [ 0.37691676, 0.46864664 ],
    [ 0.62285697, 0.46912813 ],
    [ 0.50123859, 0.61331904 ],
    [ 0.39308822, 0.72541100 ],
    [ 0.61150205, 0.72490465 ]
])


def warp_face_by_landmark(image , face_landmark_5, crop_size ):
	normed_warp_template = ffhq_512 * crop_size
	affine_matrix = cv2.estimateAffinePartial2D(face_landmark_5, normed_warp_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
	cropped = cv2.warpAffine(image, affine_matrix, crop_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
	return cropped, affine_matrix


def create_box_mask(crop_size, face_mask_blur, face_mask_padding):
    blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
    blur_area = max(blur_amount // 2, 1)
    box_mask = np.ones(crop_size, np.float32)
    box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
    box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
    box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
    box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
    if blur_amount > 0:
        box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
    return box_mask

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
