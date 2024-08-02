
import os
import torch
import yaml
import comfy.utils
import numpy as np
import tempfile
from pathlib import Path

MAX_RESOLUTION = 16384
USE_SYMLINKS = False

here = Path(__file__).parent.resolve()

config_path = Path(here, "config.yaml")

ANNOTATOR_CKPTS_PATH = ""
TEMP_DIR = ""
USE_SYMLINKS = False
ORT_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider", "OpenVINOExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider", "CoreMLExecutionProvider"]

print(f'here: {here}')

if os.path.exists(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    ANNOTATOR_CKPTS_PATH = str(Path(here, config["annotator_ckpts_path"]))
    TEMP_DIR = str(Path(here, config["custom_temp_path"]).resolve())
    USE_SYMLINKS = config["USE_SYMLINKS"]
    ORT_PROVIDERS = config["EP_list"]

    if TEMP_DIR is None:
        TEMP_DIR = tempfile.gettempdir()
    elif not os.path.isdir(TEMP_DIR):
        try:
            os.makedirs(TEMP_DIR)
        except:
            print(f"Failed to create custom temp directory. Using default.")
            TEMP_DIR = tempfile.gettempdir()
    
    if not os.path.isdir(ANNOTATOR_CKPTS_PATH):
        try:
            os.makedirs(ANNOTATOR_CKPTS_PATH)
        except:
            print(f"Failed to create config ckpts directory. Using default.")
            ANNOTATOR_CKPTS_PATH = str(Path(here, "./ckpts"))
else:
    ANNOTATOR_CKPTS_PATH = str(Path(here, "./ckpts"))
    TEMP_DIR = tempfile.gettempdir()
    USE_SYMLINKS = False
    ORT_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider", "OpenVINOExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider", "CoreMLExecutionProvider"]

os.environ['AUX_ANNOTATOR_CKPTS_PATH'] = os.getenv('AUX_ANNOTATOR_CKPTS_PATH', ANNOTATOR_CKPTS_PATH)
os.environ['AUX_TEMP_DIR'] = os.getenv('AUX_TEMP_DIR', str(TEMP_DIR))
os.environ['AUX_USE_SYMLINKS'] = os.getenv('AUX_USE_SYMLINKS', str(USE_SYMLINKS))
os.environ['AUX_ORT_PROVIDERS'] = os.getenv('AUX_ORT_PROVIDERS', str(",".join(ORT_PROVIDERS)))

print(f"Using ckpts path: {ANNOTATOR_CKPTS_PATH}")
print(f"Using symlinks: {USE_SYMLINKS}")
print(f"Using ort providers: {ORT_PROVIDERS}")    

def common_annotator_call(model, tensor_image, input_batch=False, show_pbar=True, **kwargs):
    if "detect_resolution" in kwargs:
        del kwargs["detect_resolution"] #Prevent weird case?

    if "resolution" in kwargs:
        detect_resolution = kwargs["resolution"] if type(kwargs["resolution"]) == int and kwargs["resolution"] >= 64 else 512
        del kwargs["resolution"]
    else:
        detect_resolution = 512

    if input_batch:
        np_images = np.asarray(tensor_image * 255., dtype=np.uint8)
        np_results = model(np_images, output_type="np", detect_resolution=detect_resolution, **kwargs)
        return torch.from_numpy(np_results.astype(np.float32) / 255.0)

    batch_size = tensor_image.shape[0]
    if show_pbar:
        pbar = comfy.utils.ProgressBar(batch_size)
    out_tensor = None
    for i, image in enumerate(tensor_image):
        np_image = np.asarray(image.cpu() * 255., dtype=np.uint8)
        np_result = model(np_image, output_type="np", detect_resolution=detect_resolution, **kwargs)
        out = torch.from_numpy(np_result.astype(np.float32) / 255.0)
        if out_tensor is None:
            out_tensor = torch.zeros(batch_size, *out.shape, dtype=torch.float32)
        out_tensor[i] = out
        if show_pbar:
            pbar.update(1)
    return out_tensor

def create_node_input_types(**extra_kwargs):
    return {
        "required": {
            "image": ("IMAGE",)
        },
        "optional": {
            **extra_kwargs,
            "resolution": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64})
        }
    }
