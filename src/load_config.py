from yaml import load, Loader

def load_config():
    with open('./config.yaml', 'r') as f:
        return load(f, Loader=Loader)
