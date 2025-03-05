import os
import llama
from llama import get_models_path, get_model_files, get_style_presets, get_llama_cpp_chat_handlers
from llama import LLamaModel

class LLamaOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        value =llama.LLamaOptions()
        result = {}

        for key in value:
            if type(value[key]) == bool:
                result[key] = ([True, False], {"default": value[key]})
            elif type(value[key]) == int:
                result[key] = ("INT", {
                               "default": value[key], "min": -0xffffffffffffffff, "max": 0xffffffffffffffff})
            elif type(value[key]) == float:
                result[key] = ("FLOAT", {
                               "default": value[key], "min": -0xffffffffffffffff, "max": 0xffffffffffffffff})
            elif type(value[key]) == str:
                result[key] = ("STRING", {"default": value[key]})
            elif type(value[key]) == list:
                result[key] = (value[key], {"default": value[key][0]})
            else:
                raise Exception(f"Unknown type: {type(value[key])}")

        return {
            "required": result,
        }

    RETURN_TYPES = ("LLamaOptions",)
    RETURN_NAMES = ("llama_options",)

    FUNCTION = "create"
    CATEGORY = f"tbox/Prompt"

    def create(self, **kwargs):
        kwargs = kwargs.copy()
        opt = {}
        for key in kwargs:
            opt[key] = kwargs[key]
        return (opt,)



class LLamaModelNode:
    @classmethod
    def INPUT_TYPES(s):
        model_files = get_model_files()
        chat_format = get_llama_cpp_chat_handlers()
        print(f'model_files: {model_files}')
        return {
            "required": {
                "llama_model": (model_files,),
                "chat_format": (["auto"] + chat_format, {"default": "auto"}),
            },
            "optional": {
                "options": ("LLamaOptions", ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = f"tbox/Prompt"
    
    def load(self, llama_model, chat_format='auto', options=None):
        model_path = os.path.join(get_models_path(), llama_model)
        if options == None :
            options =llama.LlamaOptions()
       
        model = LLamaModel(model_path, options)
        return (model, chat_format,)
        
        
class LLamaCLIPTextEncodeNode:
    @classmethod
    def INPUT_TYPES(s):
        style_presets = get_style_presets()
        return {
            "required": {
                "model": ("MODEL",),
                "text": ("STRING", {"multiline": True, }),
                "style": (
                    style_presets, {"default": style_presets[1]}
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "clip": ("CLIP", ),
            },
        }

    RETURN_TYPES = ("STRING", "CONDITIONING",)
    RETURN_NAMES = ("text", "conditioning",)
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = f"tbox/Prompt"


    def encode(self, model, text, style, seed, clip):

        if model == None:
            raise ValueError("llama model is None")
        
        return model.encode(clip, style, seed, text)

