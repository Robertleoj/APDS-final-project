from datasets import Data
from load_config import load_config
from utils import model_from_config
from trainer.trainer import Trainer
from sys import argv

config = load_config()

data = Data(config)

train_set = data.get_train()
val_set = data.get_val()

net = model_from_config(config)

trainer = Trainer(config=config, net=net, train_set=train_set, val_set=val_set)

n_epochs = 5 if len(argv) == 1 else int(argv[1])

trainer.train(n_epochs)