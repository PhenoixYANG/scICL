import json

def get_config(config_dir):
    with open(config_dir,'r') as fr:
        return json.load(fr)
