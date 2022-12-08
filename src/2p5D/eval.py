import os


f = open("eval.txt", 'a')

def p(stmt, fp):
    print(stmt, file=fp)
    fp.flush()
    os.fsync(fp.fileno())

p('start', f)

p('importing evaluator', f)
from evaluator import Evaluator
p('importing from utils', f)
from utils import model_from_config 
p('importing from load_configr', f)
from load_config import load_config
p('importing json', f)
import json

p("Loading config...", f)

config = load_config()

p("Loading model...", f)

net = model_from_config(config)

p("Instantiating Evaluator ...", f)


eval = Evaluator(config, device='cuda', net=net)

p("Starting to evaluate checkpoints",f)

dice = eval.evaluate_all_checkpoints()

with open('checkpoint_evaluations.json', 'w') as f2:
    json.dump(dice, f2, indent=2)

f.close()
