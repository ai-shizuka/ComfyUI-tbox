
import os 
import json

config = None

def is_logging_enabled():
    config = get_extension_config()
    if "logging" not in config:
        return False
    return config["logging"]


def log(message, type=None, always=False):
    if not always and not is_logging_enabled():
        return
    if type is not None:
        message = f"[{type}] {message}"
    print(f"WD14Tagger: {message}")
    
def get_extension_config(reload=False):
    global config
    if reload == False and config is not None:
        return config

    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    print(f'config_path: {config_path}')
    
    if not os.path.exists(config_path):
        log("Missing config.json, this extension may not work correctly. Please reinstall the extension.",
            type="ERROR", always=True)

        return {"name": "Unknown", "version": -1}
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    print(f'config: {config}')
    return config
