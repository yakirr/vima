import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as F
import random, time, os
import pandas as pd
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
from IPython import display
from . import vis as tv
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

def reconstruction_loss(x_true, x_pred : Tensor, per_sample: bool=False):
    x_true, _ = x_true
    sse = torch.sum((x_pred - x_true)**2, dim=(1,2,3))
    sse /= (x_true.shape[1]*x_true.shape[2]*x_true.shape[3])
    
    if per_sample:
        return sse
    else:
        return torch.mean(sse)

def kl_loss(mean : Tensor, logvar : Tensor):
    return -0.5 * torch.mean(
        torch.sum(1 + logvar - mean.pow(2) - logvar.exp(),
        dim=tuple(range(1, mean.dim()))))
        # dim=1)) 

def per_batch_logging(model : nn.Module, batch_num : int, rlosses : list, vaelosses : list,
        kl_weight : float, log_interval : int, scheduler : LRScheduler, epoch_start_time : int):
    lr = scheduler.get_last_lr()[0]
    cur_rloss = np.mean(rlosses[-log_interval:])
    cur_vaeloss = np.mean(vaelosses[-log_interval:])
    time_per_batch = (time.time() - epoch_start_time) / batch_num

    print(f'batch {batch_num:5d} | '
            f'lr {lr:.2g} | '
            f'r-loss {cur_rloss:.2f} | '
            f'vae-loss {cur_vaeloss:.2f} | '
            f'kl-weight {kl_weight} | '
            f'time {time_per_batch:.2f} sec')

def train_one_epoch(model : nn.Module, train_dataset : Dataset,
        optimizer : torch.optim.Optimizer, scheduler : LRScheduler,
        batch_size : int, log_interval : int=20, kl_weight : float=1,
        per_batch_logging=per_batch_logging):
    model.train()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=torch.get_default_device()))
    print(f'#batches: {len(train_loader)}')

    epoch_start_time = time.time()
    losses = []; vaelosses = []; rlosses = []
    for n, batch in enumerate(train_loader):
        # Forward pass
        predictions, mean, logvar = model.forward(batch)

        rloss = reconstruction_loss(batch, predictions)
        vaeloss = kl_weight * kl_loss(mean, logvar)
        loss = vaeloss + rloss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # Save info
        losses.append(float(loss))
        rlosses.append(float(rloss))
        vaelosses.append(float(vaeloss))

        # Log
        if n % log_interval == 0 and n > 0:
            per_batch_logging(model, n, rlosses, vaelosses, kl_weight,
                log_interval, scheduler, epoch_start_time)

    return pd.DataFrame({'loss':losses, 'rloss':rlosses, 'vaeloss':vaelosses, 'kl_weight':kl_weight})

def evaluate(model : nn.Module, eval_dataset : Dataset, batch_size : int=1000,
        detailed : bool=False, subset=None):
    if subset is not None:
        eval_dataset = torch.utils.data.Subset(eval_dataset, subset)

    model.eval()
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False)

    rlosses = []; embeddings = []
    with torch.no_grad():
        for batch in pb(eval_loader):
            predictions, mean, _ = model.forward(batch, sample_from_latent=False)

            rlosses.append(
                reconstruction_loss(batch, predictions, per_sample=True).detach().cpu().numpy()
                )
            if detailed: embeddings.append(mean.detach().cpu().numpy())

    if detailed:
        return np.concatenate(rlosses), np.concatenate(embeddings)
    else:
        return np.concatenate(rlosses).mean()

def simple_per_epoch_logging(model, val_dataset, epoch, epoch_start_time, rlosses, losslog):
    print(f'end of epoch {epoch}: avg val loss = {rlosses.mean()}')

def detailed_per_epoch_logging(model, val_dataset, epoch, epoch_start_time, rlosses, losslog, Pmin=None, Pmax=None):
    display.clear_output()
    plt.figure(figsize=(9,3))
    plt.subplot(1,2,1)
    plt.plot(losslog.loss, label='total loss', alpha=0.5)
    plt.plot(losslog.rloss, label='recon. loss', alpha=0.5)
    plt.scatter(losslog.index, losslog.val_rloss, marker='x', label='recon. loss (val)', color='green')
    plt.scatter(np.argmin(losslog.val_rloss), losslog.val_rloss.min(), color='red')
    plt.legend()
    plt.ylim(0, 1.1*np.percentile(losslog.loss.values, 95))
    plt.subplot(1,2,2)
    plt.hist(rlosses, bins=50)
    plt.show()
    print(f'epoch {epoch}. best validation reconstruction error = {losslog.val_rloss.min()}')
    print(f'\ttotal time: {time.time() - epoch_start_time}')
    ix = np.argsort(rlosses)
    examples = val_dataset[list(ix[::len(ix)//12])]
    examples = (examples[0].permute(0,2,3,1), examples[1])
    tv.plot_with_reconstruction(model, examples, channels=range(examples[0].shape[-1]), pmin=Pmin, pmax=Pmax)

def full_training(model : nn.Module, train_dataset : Dataset,
        val_dataset : Dataset, optimizer : torch.optim.Optimizer,
        scheduler : LRScheduler, batch_size : int=128, n_epochs : int=10,
        kl_weight : float=1, kl_warmup : bool=False,
        per_epoch_logging=simple_per_epoch_logging,
        per_batch_logging=per_batch_logging, per_epoch_kwargs={}):
    best_val_loss = float('inf')
    losslogs = []

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, n_epochs + 1):
            epoch_start_time = time.time()
            losslog = train_one_epoch(
                model, train_dataset, optimizer, scheduler, batch_size, kl_weight=kl_weight if kl_warmup else kl_weight * min((epoch-0) / 5, 1),
                per_batch_logging=per_batch_logging)
            rlosses, _ = evaluate(model, val_dataset, detailed=True, subset=range(0, len(val_dataset), max(1, len(val_dataset)//2000)))
            scheduler.step()

            losslog['val_rloss'] = np.nan
            losslog.val_rloss.values[-1] = rlosses.mean()
            losslogs.append(losslog)
            losslogs_sofar = pd.concat(losslogs, axis=0).reset_index(drop=True)

            per_epoch_logging(model, val_dataset, epoch, epoch_start_time, rlosses,
                losslogs_sofar, **per_epoch_kwargs)

            if rlosses.mean() < best_val_loss:
                best_val_loss = rlosses.mean()
                torch.save(model.state_dict(), best_model_params_path)

        model.load_state_dict(torch.load(best_model_params_path)) # load best model states
    return model, losslogs_sofar

def train_test_split(P, breakdown=[0.8,0.2], seed=0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    P.pytorch_mode()
    P.augmentation_on()
    return torch.utils.data.random_split(P, breakdown, generator=torch.Generator(device=torch.get_default_device()))
