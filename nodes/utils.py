import hashlib
import os
import torch
import nodes
import server
import folder_paths
import numpy as np
from typing import Iterable
from PIL import Image

BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)

DIMMAX = 8192


def tensor_to_int(tensor, bits):
    #TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)
def tensor2pil(x):
    return Image.fromarray(np.clip(255. * x.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def is_url(url):
    return url.split("://")[0] in ["http", "https"]

def strip_path(path):
    #This leaves whitespace inside quotes and only a single "
    #thus ' ""test"' -> '"test'
    #consider path.strip(string.whitespace+"\"")
    #or weightier re.fullmatch("[\\s\"]*(.+?)[\\s\"]*", path).group(1)
    path = path.strip()
    if path.startswith("\""):
        path = path[1:]
    if path.endswith("\""):
        path = path[:-1]
    return path
def hash_path(path):
    if path is None:
        return "input"
    if is_url(path):
        return "url"
    return calculate_file_hash(strip_path(path))

# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    #Larger video files were taking >.5 seconds to hash even when cached,
    #so instead the modified time from the filesystem is used as a hash
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()

def is_safe_path(path):
    if "VHS_STRICT_PATHS" not in os.environ:
        return True
    basedir = os.path.abspath('.')
    try:
        common_path = os.path.commonpath([basedir, path])
    except:
        #Different drive on windows
        return False
    return common_path == basedir

def validate_path(path, allow_none=False, allow_url=True):
    if path is None:
        return allow_none
    if is_url(path):
        #Probably not feasible to check if url resolves here
        if not allow_url:
            return "URLs are unsupported for this path"
        return is_safe_path(path)
    if not os.path.isfile(strip_path(path)):
        return "Invalid file path: {}".format(path)
    return is_safe_path(path)

def common_annotator_call(model, tensor_image, input_batch=False, show_pbar=False, **kwargs):
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

    out_tensor = None
    for i, image in enumerate(tensor_image):
        np_image = np.asarray(image.cpu() * 255., dtype=np.uint8)
        np_result = model(np_image, output_type="np", detect_resolution=detect_resolution, **kwargs)
        out = torch.from_numpy(np_result.astype(np.float32) / 255.0)
        if out_tensor is None:
            out_tensor = torch.zeros(batch_size, *out.shape, dtype=torch.float32)
        out_tensor[i] = out

    return out_tensor

def create_node_input_types(**extra_kwargs):
    return {
        "required": {
            "image": ("IMAGE",)
        },
        "optional": {
            **extra_kwargs,
            "resolution": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 64})
        }
    }


prompt_queue = server.PromptServer.instance.prompt_queue
def requeue_workflow_unchecked():
    """Requeues the current workflow without checking for multiple requeues"""
    currently_running = prompt_queue.currently_running
    print(f'requeue_workflow_unchecked >>>>>> ')
    (_, _, prompt, extra_data, outputs_to_execute) = next(iter(currently_running.values()))

    #Ensure batch_managers are marked stale
    prompt = prompt.copy()
    for uid in prompt:
        if prompt[uid]['class_type'] == 'BatchManager':
            prompt[uid]['inputs']['requeue'] = prompt[uid]['inputs'].get('requeue',0)+1

    #execution.py has guards for concurrency, but server doesn't.
    #TODO: Check that this won't be an issue
    number = -server.PromptServer.instance.number
    server.PromptServer.instance.number += 1
    prompt_id = str(server.uuid.uuid4())
    prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
    print(f'requeue_workflow_unchecked <<<<<<<<<< prompt_id:{prompt_id}, number:{number}')

requeue_guard = [None, 0, 0, {}]
def requeue_workflow(requeue_required=(-1,True)):
    assert(len(prompt_queue.currently_running) == 1)
    global requeue_guard
    (run_number, _, prompt, _, _) = next(iter(prompt_queue.currently_running.values()))
    print(f'requeue_workflow >> run_number:{run_number}\n')
    if requeue_guard[0] != run_number:
        #Calculate a count of how many outputs are managed by a batch manager
        managed_outputs=0
        for bm_uid in prompt:
            if prompt[bm_uid]['class_type'] == 'BatchManager':
                for output_uid in prompt:
                    if prompt[output_uid]['class_type'] in ["VideoSaver"]:
                        for inp in prompt[output_uid]['inputs'].values():
                            if inp == [bm_uid, 0]:
                                managed_outputs+=1
        requeue_guard = [run_number, 0, managed_outputs, {}]
    requeue_guard[1] = requeue_guard[1]+1
    requeue_guard[3][requeue_required[0]] = requeue_required[1]
    if requeue_guard[1] == requeue_guard[2] and max(requeue_guard[3].values()):
        requeue_workflow_unchecked()
