

#from .utils import get_style_presets
#from .llama_cpp import get_llama_cpp_chat_handlers
from llama.llama_cpp import LLamaOptions
from llama.llama_cpp import LLamaModel


def main(text):
    model_path = "../../../../models/gguf/Llama-3.2-3B-Instruct-Q4_K_S.gguf" 
    options = LLamaOptions()
    model = LLamaModel(model_path, options)


    resp, _ = model.encode(None, 'high_quality', 123456789, text)
    print(f'resp: {resp}')


if __name__ == "__main__":
    main('一个上身裸体下身穿JK裙子的年轻女孩子骑自行车')
