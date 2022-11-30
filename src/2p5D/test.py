from datasets import Data
from load_config import load_config


config = load_config()

data = Data(config, nuke_cache = True)