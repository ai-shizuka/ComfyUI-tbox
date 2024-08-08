
import os
import cv2
import torch
import requests
import itertools
import folder_paths
import psutil
import numpy as np
from comfy.utils import common_upscale
from io import BytesIO
from PIL import Image, ImageSequence, ImageOps
from .ffmpeg import lazy_get_audio, video_extensions
from ..utils import BIGMAX, DIMMAX, strip_path, validate_path




def is_gif(filename) -> bool:
    file_parts = filename.split('.')
    return len(file_parts) > 1 and file_parts[-1] == "gif"

def target_size(width, height, force_size, custom_width, custom_height, downscale_ratio=8) -> tuple[int, int]:
    if force_size == "Disabled":
        pass
    elif force_size == "Custom Width" or force_size.endswith('x?'):
        height *= custom_width/width
        width = custom_width
    elif force_size == "Custom Height" or force_size.startswith('?x'):
        width *= custom_height/height
        height = custom_height
    else:
        width = custom_width
        height = custom_height
    width = int(width/downscale_ratio + 0.5) * downscale_ratio
    height = int(height/downscale_ratio + 0.5) * downscale_ratio
    return (width, height)

def cv_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                       select_every_nth):
    video_cap = cv2.VideoCapture(strip_path(video))
    if not video_cap.isOpened():
        raise ValueError(f"{video} could not be loaded with cv.")

    # extract video metadata
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # set video_cap to look at start_index frame
    total_frame_count = 0
    total_frames_evaluated = -1
    frames_added = 0
    base_frame_time = 1 / fps
    prev_frame = None

    if force_rate == 0:
        target_frame_time = base_frame_time
    else:
        target_frame_time = 1/force_rate

    yield (width, height, fps, duration, total_frames, target_frame_time)


    time_offset=target_frame_time - base_frame_time
    while video_cap.isOpened():
        if time_offset < target_frame_time:
            is_returned = video_cap.grab()
            # if didn't return frame, video has ended
            if not is_returned:
                break
            time_offset += base_frame_time
        if time_offset < target_frame_time:
            continue
        time_offset -= target_frame_time
        # if not at start_index, skip doing anything with frame
        total_frame_count += 1
        if total_frame_count <= skip_first_frames:
            continue
        else:
            total_frames_evaluated += 1

        # if should not be selected, skip doing anything with frame
        if total_frames_evaluated%select_every_nth != 0:
            continue

        # opencv loads images in BGR format (yuck), so need to convert to RGB for ComfyUI use
        # follow up: can videos ever have an alpha channel?
        # To my testing: No. opencv has no support for alpha
        unused, frame = video_cap.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert frame to comfyui's expected format
        # TODO: frame contains no exif information. Check if opencv2 has already applied
        frame = np.array(frame, dtype=np.float32)
        torch.from_numpy(frame).div_(255)
        if prev_frame is not None:
            inp  = yield prev_frame
            if inp is not None:
                #ensure the finally block is called
                return
        prev_frame = frame
        frames_added += 1
        
        # if cap exists and we've reached it, stop processing frames
        if frame_load_cap > 0 and frames_added >= frame_load_cap:
            break

    if prev_frame is not None:
        yield prev_frame
        
def batched(it, n):
    while batch := tuple(itertools.islice(it, n)):
        yield batch
        
def load_video_cv(video: str, force_rate: int, force_size: str,
                  custom_width: int,custom_height: int, frame_load_cap: int,
                  skip_first_frames: int, select_every_nth: int,
                  memory_limit_mb=None):

    gen = cv_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                                select_every_nth, )
    (width, height, fps, duration, total_frames, target_frame_time) = next(gen)

    print(f'[{width}x{height}]@{fps} - duration:{duration}, total_frames: {total_frames}')
    memory_limit = None
    if memory_limit_mb is not None:
        memory_limit *= 2 ** 20
    else:
        #TODO: verify if garbage collection should be performed here.
        #leaves ~128 MB unreserved for safety
        try:
            memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - 2 ** 27
        except:
            print("Failed to calculate available memory. Memory load limit has been disabled")
    if memory_limit is not None:
  
        #TODO: use better estimate for when vae is not None
        #Consider completely ignoring for load_latent case?
        max_loadable_frames = int(memory_limit//(width*height*3*(.1)))
      
        original_gen = gen
        gen = itertools.islice(gen, max_loadable_frames)
        
    downscale_ratio = 8
    frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
    if force_size != "Disabled":
        new_size = target_size(width, height, force_size, custom_width, custom_height, downscale_ratio)
        if new_size[0] != width or new_size[1] != height:
            def rescale(frame):
                s = torch.from_numpy(np.fromiter(frame, np.dtype((np.float32, (height, width, 3)))))
                s = s.movedim(-1,1)
                s = common_upscale(s, new_size[0], new_size[1], "lanczos", "center")
                return s.movedim(1,-1).numpy()
            gen = itertools.chain.from_iterable(map(rescale, batched(gen, frames_per_batch)))
    else:
        new_size = width, height

    #Some minor wizardry to eliminate a copy and reduce max memory by a factor of ~2
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (new_size[1], new_size[0], 3)))))
    if  memory_limit is not None:
        try:
            next(original_gen)
            raise RuntimeError(f"Memory limit hit after loading {len(images)} frames. Stopping execution.")
        except StopIteration:
            pass
    if len(images) == 0:
        raise RuntimeError("No frames generated")

    #Setup lambda for lazy audio capture
    audio = lazy_get_audio(video, skip_first_frames * target_frame_time,
                               frame_load_cap*target_frame_time*select_every_nth)
    #Adjust target_frame_time for select_every_nth
    target_frame_time *= select_every_nth
    video_info = {
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1/target_frame_time,
        "loaded_frame_count": len(images),
        "loaded_duration": len(images) * target_frame_time,
        "loaded_width": new_size[0],
        "loaded_height": new_size[1],
    }

    return (images, len(images), audio, video_info)



class LoadVideoNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"default": "/Users/wadahana/Desktop/live-motion2.mp4", "multiline": True, "vhs_path_extensions": video_extensions}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                "custom_width": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                "custom_height": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
        }

    CATEGORY = "tbox/Video"

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")

    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        if kwargs['video'] is None :
            raise Exception("video is not a valid path: " + kwargs['video'])
        
        kwargs['video'] = kwargs['video'].split('\n')[0]
        
        if validate_path(kwargs['video']) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        # if is_url(kwargs['video']):
        #     kwargs['video'] = try_download_video(kwargs['video']) or kwargs['video']
        return load_video_cv(**kwargs)