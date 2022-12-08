from evaluator import Evaluator
from utils import model_from_config 
from load_config import load_config
import json


config = load_config()
net = model_from_config(config)
eval = Evaluator(config, device='cuda', net=net)

dice = eval.evaluate_all_checkpoints()

with open('checkpoint_evaluations.json', 'w') as f:
    json.dump(dice, f, indent=2)
