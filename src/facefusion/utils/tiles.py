import cv2
import numpy as np

def split_into_tiles(image, total_tiles, model_size):
    image = image.reshape(model_size[0], total_tiles, model_size[1], total_tiles, 3)
    image = image.transpose(1, 3, 0, 2, 4).reshape(total_tiles ** 2, model_size[0], model_size[1], 3)
    return image

def merge_from_tiles(image, total_tiles, model_size, dsize):
    image = np.stack(image, axis = 0).reshape(total_tiles, total_tiles, model_size[0], model_size[1], 3)
    image = image.transpose(2, 0, 3, 1, 4).reshape(dsize[0], dsize[1], 3)
    return image