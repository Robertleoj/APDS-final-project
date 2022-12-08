from evaluator import Evaluator
from utils import model_from_config 
from load_config import load_config
import json


print("Loading config...")
config = load_config()

print("Loading model...")
net = model_from_config(config)

print("Instantiating Evaluator ...")
eval = Evaluator(config, device='cuda', net=net)

print("Starting to evaluate checkpoints")
dice = eval.evaluate_all_checkpoints()

with open('checkpoint_evaluations.json', 'w') as f:
    json.dump(dice, f, indent=2)
