# configuration_manager.py
import yaml

config = None

def load_config(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config