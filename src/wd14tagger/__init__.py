import csv
import os
import sys
import asyncio
import aiohttp
import comfy.utils
import numpy as np
import onnxruntime as ort
import folder_paths
from tqdm import tqdm
from onnxruntime import InferenceSession
from PIL import Image
from server import PromptServer
from aiohttp import web
from .utils import config, log, get_extension_config


defaults = {
    "model": "wd-v1-4-moat-tagger-v2",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "replace_underscore": False,
    "trailing_comma": False,
    "exclude_tags": "",
    "ortProviders": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "HF_ENDPOINT": "https://huggingface.co"
}

config = get_extension_config()

defaults.update(config.get("settings", {}))

known_models = list(config["models"].keys())

models_dir = os.path.join(folder_paths.models_dir, "wd14_tagger") 
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def wait_for_async(async_fn, loop=None):
    return asyncio.run(async_fn())

def get_installed_models():
    models = filter(lambda x: x.endswith(".onnx"), os.listdir(models_dir))
    models = [m for m in models if os.path.exists(os.path.join(models_dir, os.path.splitext(m)[0] + ".csv"))]
    return models

async def tag(image, model_name, providers, threshold=0.35, character_threshold=0.85, exclude_tags="", replace_underscore=True, trailing_comma=False, client_id=None, node=None):
    if model_name.endswith(".onnx"):
        model_name = model_name[0:-5]
    # installed = list(get_installed_models())
    # if not any(model_name + ".onnx" in s for s in installed):
    #     await download_model(model_name, client_id, node)
    
    print(f'providers: {providers}')
    name = os.path.join(models_dir, model_name + ".onnx")
    model = InferenceSession(name, providers=providers)

    input = model.get_inputs()[0]
    height = input.shape[1]

    # Reduce to max size and pad with white
    ratio = float(height)/max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    # Read all tags from csv and locate start of each category
    tags = []
    general_index = None
    character_index = None
    with open(os.path.join(models_dir, model_name + ".csv")) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if general_index is None and row[2] == "0":
                general_index = reader.line_num - 2
            elif character_index is None and row[2] == "4":
                character_index = reader.line_num - 2
            if replace_underscore:
                tags.append(row[1].replace("_", " "))
            else:
                tags.append(row[1])

    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

    result = list(zip(tags, probs[0]))

    # rating = max(result[:general_index], key=lambda x: x[1])
    general = [item for item in result[general_index:character_index] if item[1] > threshold]
    character = [item for item in result[character_index:] if item[1] > character_threshold]

    all = character + general
    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all = [tag for tag in all if tag[0] not in remove]

    res = ("" if trailing_comma else ", ").join((item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for item in all))

    print(res)
    return res

async def download_model(model, client_id, node):
    hf_endpoint = os.getenv("HF_ENDPOINT", defaults["HF_ENDPOINT"])
    if not hf_endpoint.startswith("https://"):
        hf_endpoint = f"https://{hf_endpoint}"
    if hf_endpoint.endswith("/"):
        hf_endpoint = hf_endpoint.rstrip("/")

    url = config["models"][model]
    url = url.replace("{HF_ENDPOINT}", hf_endpoint)
    url = f"{url}/resolve/main/"
    async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
        async def update_callback(perc):
            pass
            # nonlocal client_id
            # message = ""
            # if perc < 100:
            #     message = f"Downloading {model}"
            # update_node_status(client_id, node, message, perc)

        try:
            await download_to_file(
                f"{url}model.onnx", os.path.join(models_dir,f"{model}.onnx"), update_callback, session=session)
            await download_to_file(
                f"{url}selected_tags.csv", os.path.join(models_dir,f"{model}.csv"), update_callback, session=session)
        except aiohttp.client_exceptions.ClientConnectorError as err:
            log("Unable to download model. Download files manually or try using a HF mirror/proxy website by setting the environment variable HF_ENDPOINT=https://.....", "ERROR", True)
            raise

        #update_node_status(client_id, node, None)

    return web.Response(status=200)

async def download_to_file(url, destination, update_callback, is_ext_subpath=True, session=None):
    close_session = False
    if session is None:
        close_session = True
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        session = aiohttp.ClientSession(loop=loop)
    try:
        proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        log("proxy:", proxy)
        proxy_auth = None
        if proxy:
            proxy_auth = aiohttp.BasicAuth(os.getenv("PROXY_USER", ""), os.getenv("PROXY_PASS", ""))

        async with session.get(url, proxy=proxy, proxy_auth=proxy_auth) as response:            
            size = int(response.headers.get('content-length', 0)) or None

            with tqdm(
                unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1], total=size,
            ) as progressbar:
                with open(destination, mode='wb') as f:
                    perc = 0
                    async for chunk in response.content.iter_chunked(2048):
                        f.write(chunk)
                        progressbar.update(len(chunk))
                        if update_callback is not None and progressbar.total is not None and progressbar.total != 0:
                            last = perc
                            perc = round(progressbar.n / progressbar.total, 2)
                            if perc != last:
                                last = perc
                                await update_callback(perc)
    finally:
        if close_session and session is not None:
            await session.close()
