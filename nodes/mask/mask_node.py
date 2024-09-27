
import torch
import numpy as np

class MaskSubNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),  
            },
            "optional": {
                "src1": ("MASK",),
                "src2": ("MASK",),
                "src3": ("MASK",),
                "src4": ("MASK",),
                "src5": ("MASK",),
                "src6": ("MASK",),
            }
        }

    CATEGORY = "mask"
    RETURN_TYPES = ("MASK",)

    FUNCTION = "sub"
    CATEGORY = "tbox/Mask"
    
    def sub_mask(self, dst, src):
        if src != None:
            mask = src.reshape((-1, src.shape[-2], src.shape[-1]))
            return dst - mask
        return dst
        
    def add(self, mask, src1=None, src2=None, src3=None, src4=None, src5=None, src6=None):
        print(f'mask shape: {mask.shape}')
        output = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()
        output[:, :, :] = self.sub_mask(output, src1)
        output[:, :, :] = self.sub_mask(output, src2)
        output[:, :, :] = self.sub_mask(output, src3)
        output[:, :, :] = self.sub_mask(output, src4)
        output[:, :, :] = self.sub_mask(output, src5)
        output[:, :, :] = self.sub_mask(output, src6)
        output = torch.clamp(output, 0.0, 1.0)
        return (output, )  

class MaskAddNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),  
            },
            "optional": {
                "src1": ("MASK",),
                "src2": ("MASK",),
                "src3": ("MASK",),
                "src4": ("MASK",),
                "src5": ("MASK",),
                "src6": ("MASK",),
            }
        }

    CATEGORY = "mask"
    RETURN_TYPES = ("MASK",)

    FUNCTION = "add"
    CATEGORY = "tbox/Mask"
    
    def add_mask(self, dst, src):
        if src != None:
            mask = src.reshape((-1, src.shape[-2], src.shape[-1]))
            return dst + mask
        return dst
        
    def add(self, mask, src1=None, src2=None, src3=None, src4=None, src5=None, src6=None):
        print(f'mask shape: {mask.shape}')
        output = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()
        output[:, :, :] = self.add_mask(output, src1)
        output[:, :, :] = self.add_mask(output, src2)
        output[:, :, :] = self.add_mask(output, src3)
        output[:, :, :] = self.add_mask(output, src4)
        output[:, :, :] = self.add_mask(output, src5)
        output[:, :, :] = self.add_mask(output, src6)
        output = torch.clamp(output, 0.0, 1.0)
        return (output, )  
    