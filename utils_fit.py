from tqdm import tqdm
import torch
import numpy as np


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_fn(model, data_loader, optimizer, device, epochs, EPOCHS):
    model.train()
    fin_loss = []
    loop = tqdm(data_loader, total=len(data_loader))
    for x, y in loop:
        x = x.to(device)
        y = [ann.to(device) for ann in y]
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        fin_loss.append(loss.item())
        loop.set_postfix(epochs=f"{epochs}/{EPOCHS}", loss=loss.item(), lr=get_lr(optimizer))
    fin_loss = np.average(fin_loss)
    return fin_loss


def eval_fn(model, data_loader, device):
    model.eval()
    fin_loss = []
    loop = tqdm(data_loader, total=len(data_loader))
    for x, y, in loop:
        with torch.no_grad():
            x = x.to(device)
            y = [ann.to(device) for ann in y]
            _, loss = model(x, y)
        fin_loss.append(loss.item())
    fin_loss = np.average(fin_loss)
    return fin_loss