import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as F
import random, time, os
import pandas as pd
from tempfile import TemporaryDirectory

from vima.data.patchcollection import PatchCollection
from .logging import LossLogger
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

def seed(seed=0, deterministic=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.mps.deterministic = True
        torch.backends.mps.benchmark = False

def reconstruction_loss(x_true, x_pred : Tensor, per_sample: bool=False):
    x_true, _ = x_true
    sse = torch.sum((x_pred - x_true)**2, dim=(1,2,3))
    sse /= (x_true.shape[1]*x_true.shape[2]*x_true.shape[3])
    
    if per_sample:
        return sse
    else:
        return torch.mean(sse)

def kl_loss(mean : Tensor, logvar : Tensor, per_sample: bool=False):
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(),
            dim=tuple(range(1, mean.dim())))
    if per_sample:
        return kl
    else:
        return torch.mean(kl)

def train_one_epoch(models : list[nn.Module], train_dataset : Dataset,
        optimizers : list[torch.optim.Optimizer], schedulers : list[LRScheduler],
        batch_size : int, log : LossLogger, kl_weight : float=1):
    for model in models:
        model.train()
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=torch.get_default_device()))
    
    print(f'#batches: {len(train_loader)}')
    for n, batch in enumerate(train_loader):
        # Forward pass
        x, sids = batch
        for modelid, (model, optimizer) in enumerate(zip(models, optimizers)):
            predictions, mean, logvar = model.forward([x, sids])
            rloss = reconstruction_loss(batch, predictions)
            vaeloss = kl_weight * kl_loss(mean, logvar)
            loss = vaeloss + rloss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # Save info and log if necessary
            log.log_batch(modelid, n, loss, rloss, vaeloss, kl_weight, optimizer.param_groups[0]['lr'])

def evaluate(model : nn.Module, eval_dataset : Dataset, kl_weight : float,
             batch_size : int=1000, detailed : bool=False, subset=None, sample_from_latent=False):
    if subset is not None:
        eval_dataset = torch.utils.data.Subset(eval_dataset, subset)

    model.eval()
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device=torch.get_default_device()))

    rlosses = []
    kllosses = []
    embeddings = []
    with torch.no_grad():
        for batch in pb(eval_loader):
            predictions, mean, logvar = model.forward(batch, sample_from_latent=sample_from_latent)

            rlosses.append(reconstruction_loss(batch, predictions, per_sample=True).detach().cpu().numpy())
            kllosses.append(kl_weight * kl_loss(mean, logvar, per_sample=True).detach().cpu().numpy())
            if detailed: embeddings.append(mean.detach().cpu().numpy())

    if detailed:
        return np.concatenate(rlosses), np.concatenate(kllosses), np.concatenate(embeddings)
    else:
        return np.concatenate(rlosses).mean(), np.concatenate(kllosses).mean()

def full_training(models : list[nn.Module], train_dataset : Dataset,
        val_dataset : Dataset, optimizers : list[torch.optim.Optimizer],
        schedulers : list[LRScheduler], P : PatchCollection, log : LossLogger,
        batch_size : int=128, n_epochs : int=10,
        kl_weight : float=1, kl_warmup : bool=True, stop_augmentation : float=1):
    best_val_losses = [float('inf') for model in models]
    best_epoch = [-1 for i in range(len(models))]

    with TemporaryDirectory() as tempdir:
        best_model_params_paths = [
            os.path.join(tempdir, f"best_model_params_{i}.pt")
            for i in range(len(models))
        ]

        for epoch in range(1, n_epochs + 1):
            if epoch > stop_augmentation * n_epochs:
                P.augmentation_off()

            train_one_epoch(
                models, train_dataset, optimizers, schedulers, batch_size, log,
                    kl_weight=kl_weight * min(epoch / 5, 1) if kl_warmup else kl_weight)
            
            for modelid, (model, scheduler, best_path) in enumerate(zip(models, schedulers, best_model_params_paths)):
                rlosses, kllosses, _ = evaluate(model, val_dataset, kl_weight,
                    detailed=True, subset=range(0, len(val_dataset), max(1, len(val_dataset)//2000)))
                scheduler.step()
                log.log_epoch(modelid, rlosses + kllosses, rlosses, kllosses, models, val_dataset)

                total_loss = rlosses.mean() + kllosses.mean()
                if total_loss < best_val_losses[modelid]:
                    best_val_losses[modelid] = total_loss
                    best_epoch[modelid] = epoch
                    torch.save(model.state_dict(), best_path)
            print('best validation losses so far across full model ensemble:', best_val_losses)
            print('best epochs so far across full model ensemble:', best_epoch)

        for model, best_path in zip(models, best_model_params_paths):
            model.load_state_dict(torch.load(best_path)) # load best model states

def train_test_split(P, breakdown=[0.8,0.2]):
    P.pytorch_mode()
    P.augmentation_on()
    return torch.utils.data.random_split(P, breakdown, generator=torch.Generator(device=torch.get_default_device()))

def train(models, P, kl_weight=1e-5, kl_warmup=True, stop_augmentation=1,
          batch_size=256, n_epochs=20, lr=1e-3, gamma=0.9,
          plot_reconstructions=False, on_epoch_end=None):
    train_dataset, val_dataset = train_test_split(P)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=lr) for model in models]
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma) for optimizer in optimizers]
    log = LossLogger(log_interval=20, detailed=plot_reconstructions,
                     Pmin=P.vmin, Pmax=P.vmax,
                     on_epoch_end=on_epoch_end)
    
    full_training(models, train_dataset, val_dataset, optimizers, schedulers, P, log,
                                    batch_size=batch_size, n_epochs=n_epochs,
                                    kl_weight=kl_weight, kl_warmup=kl_warmup,
                                    stop_augmentation=stop_augmentation)
    
    return log