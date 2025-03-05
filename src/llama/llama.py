
import json
import llama_cpp
from llama_cpp import Llama
from .utils import get_schema_base_type, get_schema_array, get_schema_obj, a1111_clip_text_encode
from .prompts import Beautify_Prompt, Long_Prompt

def LLamaOptions():
    return {
        # "chat_format": chat_handlers,
        "n_ctx": 2048,
        "n_batch": 2048,
        "n_threads": 0,
        "n_threads_batch": 0,
        "split_mode": ["LLAMA_SPLIT_MODE_NONE", "LLAMA_SPLIT_MODE_LAYER", "LLAMA_SPLIT_MODE_ROW",],
        "main_gpu": 0,
        "n_gpu_layers": -1,
        "max_tokens": 4096,
        "temperature": 1.6,
        "top_p": 0.95,
        "min_p": 0.05,
        "typical_p": 1.0,
        "stop": "",
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.1,
        "top_k": 50,
        "tfs_z": 1.0,
        "mirostat_mode": ["none", "mirostat", "mirostat_v2"],
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
        "verbose": False,
    }


class LLamaModel: 
    def __init__(self, model_file, options):
        split_mode_int = llama_cpp.LLAMA_SPLIT_MODE_LAYER
        if options.get("split_mode", "LLAMA_SPLIT_MODE_LAYER") == "LLAMA_SPLIT_MODE_ROW":
            split_mode_int = llama_cpp.LLAMA_SPLIT_MODE_ROW
        elif options.get("split_mode", "LLAMA_SPLIT_MODE_LAYER") == "LLAMA_SPLIT_MODE_NONE":
            split_mode_int = llama_cpp.LLAMA_SPLIT_MODE_NONE
            
        model = Llama(
            model_path=model_file,
            n_gpu_layers=options.get("n_gpu_layers", -1),
            n_ctx=options.get("n_ctx", 2048),
            n_batch=options.get("n_batch", 2048),
            n_threads=options.get("n_threads", 0) if options.get("n_threads", 0) > 0 else None,
            n_threads_batch=options.get("n_threads_batch", 0) if options.get("n_threads_batch", 0) > 0 else None,
            main_gpu=options.get("main_gpu", 0),
            split_mode=split_mode_int,
            logits_all=options.get("logits_all", False),
            chat_handler=None,
            chat_format=options.get("chat_format", None),
            seed=options.get("seed", -1),
            verbose=False,#options.get("verbose", False),
        )
               
        self.options = options
        self.model = model

    def encode(self, clip, style, seed, text):
        
        question = f"IDEA: {style},{text}"
        if style == "none":
            question = f"IDEA: {text}"

        system_prompt = Beautify_Prompt + Long_Prompt + "\n"

        self.model.set_seed(seed)
        self.model.reset()
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            },
        ]
        options = self.options.copy()
        response_text = self.llama_cpp_messages(messages, options)
        try:
            response_json = json.loads(response_text)
        except Exception as e:
            from . import half_json
            print("json.loads failed, try fix response_text: ", response_text)
            json_fixer = half_json.JSONFixer()
            fix_resp = json_fixer.fix(response_text)
            if fix_resp.success:
                print("fix success, use fixed response_text: ", fix_resp.line)
                response_json = json.loads(fix_resp.line)
            else:
                raise e
        responses = []
        for key, value in response_json.items():
            if type(value) == list:
                # 去除开头.和空格
                value = [v.strip().lstrip(".") for v in value]
                # 去除空字符串
                value = [v for v in value if v != ""]
                if len(value) > 0:
                    responses.append(f"({', '.join(value)})")

            else:
                if value != "":
                    responses.append(f"({value})")

        response = ", ".join(responses)
        # 去除换行
        while response.find("\n") != -1:
            response = response.replace("\n", " ")

        # 句号换成逗号
        while response.find(".") != -1:
            response = response.replace(".", ",")

        # 去除多余逗号
        while response.find(",,") != -1:
            response = response.replace(",,", ",")
        while response.find(", ,") != -1:
            response = response.replace(", ,", ",")
        conditionings = None
  
        if clip is not None:
            conditionings = a1111_clip_text_encode(
                clip, response, )

        return (response, conditionings)

    def get_response_format(self): 
        schema = get_schema_obj(keys_type={
                        "description": get_schema_base_type("string"),
                        "long_prompt": get_schema_base_type("string"),
                        "main_color_word": get_schema_base_type("string"),
                        "camera_angle_word": get_schema_base_type("string"),
                        "style_words": get_schema_array("string"),
                        "subject_words": get_schema_array("string"),
                        "light_words": get_schema_array("string"),
                        "environment_words": get_schema_array("string"),
                    },
                    required=[
                        "description",
                        "long_prompt",
                        "main_color_word",
                        "camera_angle_word",
                        "style_words",
                        "subject_words",
                        "light_words",
                        "environment_words",
                    ]
                )
        response_format = {
            "type": "json_object",
            "schema": schema,
        }
        return response_format
    
    def llama_cpp_messages(self, messages, options):
        stop = self.options.get("stop", "")
        if stop == "":
            stop = []
        else:
            # 所有转译序列
            escape_sequence = {
                "\\n": "\n",
                "\\t": "\t",
                "\\r": "\r",
                "\\b": "\b",
                "\\f": "\f",
            }
            for key, value in escape_sequence.items():
                stop = stop.replace(key, value)
            stop = stop.split(",")

        mirostat_mode = 0
        if self.options.get("mirostat_mode", "none") == "mirostat":
            mirostat_mode = 1
        elif self.options.get("mirostat_mode", "none") == "mirostat_v2":
            mirostat_mode = 2
        
        print(f'messages: {messages}')
        output = self.model.create_chat_completion(
            messages=messages,
            response_format=self.get_response_format(),
            max_tokens=options.get("max_tokens", 4096),
            temperature=options.get("temperature", 1.6),
            top_p=options.get("top_p", 0.95),
            min_p=options.get("min_p", 0.05),
            typical_p=options.get("typical_p", 1.0),
            stop=stop,
            frequency_penalty=options.get("frequency_penalty", 0.0),
            presence_penalty=options.get("presence_penalty", 0.0),
            repeat_penalty=options.get("repeat_penalty", 1.1),
            top_k=options.get("top_k", 50),
            tfs_z=options.get("tfs_z", 1.0),
            mirostat_mode=mirostat_mode,
            mirostat_tau=options.get("mirostat_tau", 5.0),
            mirostat_eta=options.get("mirostat_eta", 0.1),
            tools=options.get("tools", None),
            tool_choice=options.get("tool_choice", None),
        )
        print(f'output: {output}')
        choices = output.get("choices", [])
        if len(choices) == 0:
            return ""

        result = choices[0].get("message", {}).get("content", "")
        result = result.replace("\n", " ")
        return result
        
# if __name__ == "__main__":
        
#     model_path = '../../../../models/gguf/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf'

#     model = Llama(
#             model_path=model_file,
#             n_gpu_layers=options.get("n_gpu_layers", -1),
#             n_ctx=2048,
#             n_batch=2048,
#             n_threads=None,
#             n_threads_batch=None, 
#             main_gpu=0,
#             split_mode=LLAMA_SPLIT_MODE_NONE,
#             logits_all=options.get("logits_all", False),
#             chat_handler=chat_handler,
#             chat_format=auto,
#             seed=options.get("seed", -1),
#             verbose=verbose,
#         )
#         model_and_opt = {
#             "model": model,
#             "chat_handler": chat_handler,
#             "options": options,
#         }
    