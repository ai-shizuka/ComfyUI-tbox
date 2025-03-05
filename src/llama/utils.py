import os 
import folder_paths
from llama_cpp import llama_chat_format

def get_schema_base_type(t):
    return {
        "type": t,
    }

def get_schema_array(item_type="string"):
    if type(item_type) == str:
        item_type = get_schema_base_type(item_type)
    return {
        "type": "array",
        "items": item_type,
    }

def get_schema_obj(keys_type={}, required=[]):
    item = {}
    for key, value in keys_type.items():
        if type(value) == str:
            value = get_schema_base_type(value)
        item[key] = value
    return {
        "type": "object",
        "properties": item,
        "required": required
    }


high_quality_prompt = "((high quality:1.4), (best quality:1.4), (masterpiece:1.4), (8K resolution), (2k wallpaper))"
style_presets_prompt = {
    "none": "",
    "high_quality": high_quality_prompt,
    "photography": f"{high_quality_prompt}, (RAW photo, best quality), (realistic, photo-realistic:1.2), (bokeh, cinematic shot, dynamic composition, incredibly detailed, sharpen, details, intricate detail, professional lighting, film lighting, 35mm, anamorphic, lightroom, cinematography, bokeh, lens flare, film grain, HDR10, 8K)",
    "illustration": f"{high_quality_prompt}, ((detailed matte painting, intricate detail, splash screen, complementary colors), (detailed),(intricate details),illustration,an extremely delicate and beautiful,ultra-detailed,highres,extremely detailed)",
}

def get_style_presets():
    return [
        "none",
        "high_quality",
        "photography",
        "illustration",
    ]

def get_llama_cpp_chat_handlers():
    chat_handlers = llama_chat_format.LlamaChatCompletionHandlerRegistry()._chat_handlers
    chat_handlers = list(chat_handlers.keys())
    return chat_handlers
        

def get_models_path():
    models_path = os.path.join(folder_paths.models_dir, "gguf")
    os.makedirs(models_path, exist_ok=True)
    return models_path

def get_model_files():
    models_dir = get_models_path()
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_files = []
    # walk model_files
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".gguf"):
                model_files.append(
                    os.path.relpath(os.path.join(root, file), models_dir))
    return model_files

def a1111_clip_text_encode(clip, text):
    try:
        from . import ADV_CLIP_emb_encode
        cond, pooled = ADV_CLIP_emb_encode.advanced_encode(
            clip, text, "none", "A1111", w_max=1.0, apply_to_pooled=False)
        return [[cond, {"pooled_output": pooled}]]
    except Exception as e:
        import nodes
        return nodes.CLIPTextEncode().encode(clip, text)[0]