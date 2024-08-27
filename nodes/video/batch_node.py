
import hashlib
import os

class BatchManagerNode:
    def __init__(self, frames_per_batch=-1):
        print("BatchNode init")
        self.frames_per_batch = frames_per_batch
        self.inputs = {}
        self.outputs = {}
        self.unique_id = None
        self.has_closed_inputs = False
        self.total_frames = float('inf')
    
    def reset(self):
        print("BatchNode reset")
        self.close_inputs()
        for key in self.outputs:
            if getattr(self.outputs[key][-1], "gi_suspended", False):
                try:
                    self.outputs[key][-1].send(None)
                except StopIteration:
                    pass
        self.__init__(self.frames_per_batch)
    def has_open_inputs(self):
        return len(self.inputs) > 0
    def close_inputs(self):
        for key in self.inputs:
            if getattr(self.inputs[key][-1], "gi_suspended", False):
                try:
                    self.inputs[key][-1].send(1)
                except StopIteration:
                    pass
        self.inputs = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {"frames_per_batch": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1})},
                "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("BatchManager",)
    RETURN_NAMES = ("meta_batch",)
    CATEGORY = "tbox/Video"
    FUNCTION = "update_batch"
    
    def update_batch(self, frames_per_batch, prompt=None, unique_id=None):
       
        if unique_id is not None and prompt is not None:
            requeue = prompt[unique_id]['inputs'].get('requeue', 0)
        else:
            requeue = 0
        print(f'update_batch >> unique_id: {unique_id}; requeue: {requeue}')
        if requeue == 0:
            self.reset()
            self.frames_per_batch = frames_per_batch
            self.unique_id = unique_id
        else:
            num_batches = (self.total_frames+self.frames_per_batch-1)//frames_per_batch
            print(f'Meta-Batch {requeue}/{num_batches}')
        #onExecuted seems to not be called unless some message is sent
        return (self,)
    
    @classmethod
    def IS_CHANGED(self, frames_per_batch, prompt=None, unique_id=None):
        print(f"BatchManagerNode >>>  IS_CHANGED : {result}")
        random_bytes = os.urandom(32)
        result = hashlib.sha256(random_bytes).hexdigest()
        return result
