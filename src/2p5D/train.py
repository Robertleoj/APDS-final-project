import torch
from torch import optim,nn
from utils import dice_all



def train(*,
    net: nn.Module, 
    train_loader, 
    val_loader, 
    n_epochs, 
    device='cuda',
    optimizer: torch.optim.Optimizer,
    loss_fn,
    config
):

    mask_classes = config['mask_classes']

    train_losses = []
    train_dice = {k: [] for k in mask_classes.keys()}
    
    val_losses = []
    val_dice = {}

    iters = 0

    for ep in range(n_epochs):

        ep_train_loss = []
        ep_train_dice = []

        for X_train, y_train in train_loader:
            X_train: torch.Tensor
            y_train: torch.Tensor



            X_train, y_train = X_train.to(device), y_train.to(device, dtype=torch.long)

            out: torch.Tensor = net(X_train)

            loss: torch.Tensor = loss_fn(out, y_train)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            preds = out.argmax(1)
            train_d = dice_all(preds.detach().cpu(), y_train.detach().cpu(), mask_classes)
            ep_train_dice.append(train_d)

            train_l = loss.detach().item()
            ep_train_loss.append(train_l)

            iters += 1
            print(f"[Iter {iters}] \n\tloss: {train_l}\n\tdice: {train_d}")

        train_loss = torch.tensor(ep_train_loss).mean().item()
        train_die = {}
        for key in mask_classes.keys():
            die_mean = torch.tensor([e[key] for e in ep_train_dice]).mean().item()
            train_die[key] = die_mean
            train_dice[key].append(die_mean)

        train_losses.append(train_loss)

        print(f"[{ep + 1} / {n_epochs}]")
        print(f"\ttrain_loss: {train_loss}")
        print(f"\ttrain_dice: {train_die}")

        
        # ep_val_loss = []
        # ep_val_dice = []

        # for X_train, y_train in train_loader:
        #     X_train: torch.Tensor
        #     y_train: torch.Tensor



        #     X_train, y_train = X_train.to(device), y_train.to(device)

        #     out: torch.Tensor = net(X_train)

        #     loss: torch.Tensor = loss_fn(out, y_train)

        #     optimizer.zero_grad()

        #     loss.backward()

        #     optimizer.step()

        #     preds = out.argmax(-1)

        #     ep_train_dice.append(dice_all(preds.detach().cpu(), y_train.detach().cpu()))
        #     ep_train_loss.append(loss.detach().item())

        # train_loss = torch.tensor(ep_train_loss).mean().item()
        # train_die = {}
        # for key in mask_classes.keys():
        #     die_mean = torch.tensor([e[key] for e in ep_train_dice]).mean().item()
        #     train_die[key] = die_mean
        #     train_dice[key].append(die_mean)

        # train_losses.append(train_loss)
        






