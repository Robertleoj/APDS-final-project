import torch
from torch import optim,nn
from utils import dice_all
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from csv import DictWriter


class Trainer:

    def __init__(self, *, 
        config,
        net: nn.Module,
        train_set: Dataset,
        val_set: Dataset,
        load=True
    ):
        self.config = config

        self.train_set = train_set
        self.val_set = val_set

        self.device = config['device']
        self.checkpoint_dir = config['checkpoint_dir']
        self.save_every_iter = config['save_every_iter']
        self.mask_classes = config['mask_classes']

        self.train_log_file_iters = config['train_log_file_iters']
        self.train_log_file_epochs = config['train_log_file_epochs']

        self.log_every_iter = config['log_every_iter']
        self.log_every_epoch = config['log_every_epoch']


        self.net = net.to(self.device)

        self.total_epochs = 0
        self.total_iters = 0
        
        if load:
            self.load_newest()
        
        self.__initialize_loaders()
        self.__init_optimizer()
        self.__init_loss_function()

    def load_newest(self):
        ckpt_paths = glob(f"{self.checkpoint_dir}/*.ckpt")

        if len(ckpt_paths) == 0:
            print("No checkpoint found. Starting with a randomly initialized model.")
            return

        ckpt_tuples = []

        for path in ckpt_paths:
            splitted = path.split('_')
            epoch = int(splitted[-2])
            iter = int(splitted[-1].split('.')[0])

            ckpt_tuples.append((epoch, iter))

        epoch, iter = max(ckpt_tuples)

        self.load_checkpoint(epoch, iter)


    def load_checkpoint(self, epoch, iter):
        path = f"{self.checkpoint_dir}/weights_{epoch}_{iter}.ckpt"
        if not os.path.exists(path):
            raise ValueError(f"Checkpoint ({epoch=}, {iter=}) does not exist.")
        
        self.net.load_state_dict(
            torch.load(path, map_location=self.device)
        )

        self.total_epochs = epoch
        self.total_iters = iter
        self.last_iter_saved = iter


    def __initialize_loaders(self):
        batch_size = self.config['batch_size']
        num_workers = self.config['dataloader_num_workers']

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_set,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )

    def __save_checkpoint(self):
        fname = f"weights_{self.total_epochs}_{self.total_iters}.ckpt"
        fpath = f"{self.checkpoint_dir}/{fname}"
        torch.save(self.net.state_dict(), fpath)

    def __init_optimizer(self):
        lr = self.config['learning_rate']
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)


    def __init_loss_function(self):
        loss_weights = torch.tensor(self.config['loss_weights']).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)


    def __iterate(self, X: torch.Tensor, y: torch.Tensor, train=True):
        X = X.to(self.device)
        y = y.to(self.device)

        if train:
            self.net.train()
        else:
            self.net.eval()

        with torch.set_grad_enabled(train):
            out = self.net(X)

        loss: torch.Tensor = self.loss_fn(out, y)

        if train:
            loss.backward()
            self.optimizer.step()

        preds = out.argmax(1)

        dice = dice_all(preds.detach().cpu(), y.detach().cpu(), self.mask_classes)
        loss = loss.detach().item()

        if train:
            self.total_iters += 1
            self.optimizer.zero_grad()

        return loss, dice

    def __make_epoch_total_dice(self, epoch_dice):
        dice = {}

        for key in self.mask_classes.keys():
            die_mean = torch.tensor([e[key] for e in epoch_dice]).mean().item()
            dice[key] = die_mean

        return dice
        
    def __write_log(self, file: str, fields: dict, fieldnames):
         with open(file, 'a') as f_object:
 
            writer_object = DictWriter(f_object, fieldnames=fieldnames)

            if not os.path.exists(file):
                writer_object.writeheader()
 
            writer_object.writerow(fields)

       
    
    def __log_performance_iters(self, train_loss, train_dice):
        path = self.train_log_file_iters

        fields = {
            "epoch": self.total_epochs,
            "iter": self.total_iters,
            "train_loss": train_loss,
            "train_dice_empty": train_dice["empty"],
            "train_dice_liver": train_dice["liver"],
            "train_dice_cancer": train_dice["cancer"]
        }

        fieldnames = [
            "epoch", 
            "iter", 
            "train_loss", 
            "train_dice_empty", 
            "train_dice_liver", 
            "train_dice_cancer"
        ]

        self.__write_log(file=path,fields=fields, fieldnames=fieldnames)
        
        
    def __log_performance_epochs(self, train_loss, train_dice, val_loss, val_dice):
        path = self.train_log_file_epochs

        fieldnames = [
            'epoch', 
            'iter', 
            'train_loss', 
            'train_dice_empty', 
            'train_dice_liver', 
            'train_dice_cancer', 
            'val_loss',
            'val_dice_empty', 
            'val_dice_liver', 
            'val_dice_cancer',
        ]

        fields = {
            "epoch": self.total_epochs,
            "iter": self.total_iters,

            "train_loss": train_loss,
            "train_dice_empty": train_dice['empty'],
            "train_dice_liver": train_dice['liver'],
            "train_dice_cancer": train_dice['cancer'],

            "val_loss": val_loss,
            "val_dice_empty": val_dice['empty'],
            "val_dice_liver": val_dice['liver'],
            "val_dice_cancer": val_dice['cancer'],
        }

        self.__write_log(path, fields, fieldnames)
        

    def train(self, n_epochs):

        for _ in range(n_epochs):

            ep_train_loss = []
            ep_train_dice = []

            for X_train, y_train in self.train_loader:
                train_l, train_d = self.__iterate(X_train, y_train, train=True)
                
                if (self.total_iters % self.log_every_iter) == 0:
                    self.__log_performance_iters(train_l, train_d)

                if self.total_iters % self.save_every_iter == 0:
                    self.__save_checkpoint()

                ep_train_dice.append(train_d)
                ep_train_loss.append(train_l)

            train_loss = torch.tensor(ep_train_loss).mean().item()
            train_dice = self.__make_epoch_total_dice(ep_train_dice)
        
            ep_val_loss = []
            ep_val_dice = []

            for X_val, y_val in self.val_loader:
                val_l, val_d = self.__iterate(X_val, y_val, train=False)
                ep_val_dice.append(val_d)
                ep_val_loss.append(val_l)

            val_loss = torch.tensor(ep_train_loss).mean().item()
            val_dice = self.__make_epoch_total_dice(ep_train_dice)

            self.total_epochs += 1

            if self.total_epochs % self.log_every_epoch == 0:
                self.__log_performance_epochs(train_loss, train_dice, val_loss, val_dice)


 

