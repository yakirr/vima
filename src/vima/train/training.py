import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as F
import random, time, os, pickle
import pandas as pd
from tempfile import TemporaryDirectory

from vima.data.patchcollection import PatchCollection
from .logging import LossLogger
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

def set_seed(seed=0, deterministic=True):
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
        generator : torch.Generator,
        optimizers : list[torch.optim.Optimizer], schedulers : list[LRScheduler],
        batch_size : int, log : LossLogger, kl_weight : float=1):
    for model in models:
        model.train()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator)

    print(f'#batches: {len(train_loader)}')
    for n, batch in enumerate(train_loader):
        x, sids = batch
        losses = {}

        # Forward pass
        for modelid, (model, optimizer) in enumerate(zip(models, optimizers)):
            predictions, mean, logvar = model.forward([x, sids])
            rloss = reconstruction_loss(batch, predictions)
            vaeloss = kl_weight * kl_loss(mean, logvar)
            losses[modelid] = (vaeloss, rloss, vaeloss + rloss)

        # Backward pass
        for modelid, (model, optimizer) in enumerate(zip(models, optimizers)):
            vaeloss, rloss, loss = losses[modelid]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # Save info and log if necessary
            log.log_batch(modelid, n, loss, rloss, vaeloss, kl_weight, optimizer.param_groups[0]['lr'])

def evaluate(model : nn.Module, eval_dataset : Dataset,
             generator : torch.Generator,
             kl_weight : float,
             batch_size : int=1000, detailed : bool=False, subset=None, sample_from_latent=False):
    if subset is not None:
        eval_dataset = torch.utils.data.Subset(eval_dataset, subset)

    model.eval()
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        generator=generator)

    rlosses = []
    kllosses = []
    embeddings = []
    channel_errs = []
    with torch.no_grad():
        for batch in eval_loader:
            predictions, mean, logvar = model.forward(batch, sample_from_latent=sample_from_latent)

            rlosses.append(reconstruction_loss(batch, predictions, per_sample=True).detach().cpu().numpy())
            kllosses.append(kl_weight * kl_loss(mean, logvar, per_sample=True).detach().cpu().numpy())
            if detailed:
                embeddings.append(mean.detach().cpu().numpy())
                x_true, _ = batch
                channel_errs.append(((predictions - x_true)**2).mean(dim=(0,2,3)).detach().cpu().numpy())

    if detailed:
        return np.concatenate(rlosses), np.concatenate(kllosses), np.concatenate(embeddings), np.stack(channel_errs).mean(axis=0)
    else:
        return np.concatenate(rlosses).mean(), np.concatenate(kllosses).mean()


class TrainingCheckpoint:
    """Owns all mutable training state and handles checkpoint save/resume."""

    def __init__(self, models, optimizers, schedulers, log,
                 best_model_params_paths, hyperparams: dict):
        self.models = models
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.log = log
        self.best_model_params_paths = best_model_params_paths
        self.hyperparams = hyperparams

        self.best_val_losses = [float('inf')] * len(models)
        self.best_epoch      = [-1] * len(models)
        self.resume_epoch    = 1

    def save(self, checkpoint_dir: str, epoch: int, generator: torch.Generator):
        # Pickle log without the callback (may capture unpicklable closures)
        saved_cb = self.log.on_epoch_end
        self.log.on_epoch_end = None
        log_bytes = pickle.dumps(self.log)
        self.log.on_epoch_end = saved_cb

        state = {
            'resume_epoch':      epoch + 1,
            'best_val_losses':   self.best_val_losses,
            'best_epoch':        self.best_epoch,
            'hyperparams':       self.hyperparams,
            'generator_state':   generator.get_state(),
            'model_states':      [m.state_dict() for m in self.models],
            'optimizer_states':  [o.state_dict() for o in self.optimizers],
            'scheduler_states':  [s.state_dict() for s in self.schedulers],
            'best_model_states': [
                torch.load(p, weights_only=True)
                for p in self.best_model_params_paths
                if os.path.exists(p)
            ],
            'log': log_bytes,
        }
        tmp = os.path.join(checkpoint_dir, 'checkpoint.pt.tmp')
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(checkpoint_dir, 'checkpoint.pt'))

        for i, bepoch in enumerate(self.best_epoch):
            if bepoch == epoch:
                torch.save(self.models[i].state_dict(),
                           os.path.join(checkpoint_dir, f'best_model_{i}.pt'))

    def load(self, checkpoint_dir: str, generator: torch.Generator):
        state = torch.load(
            os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=False)
        self.resume_epoch    = state['resume_epoch']
        self.best_val_losses = state['best_val_losses']
        self.best_epoch      = state['best_epoch']
        generator.set_state(state['generator_state'])
        for m, s in zip(self.models,      state['model_states']):     m.load_state_dict(s)
        for o, s in zip(self.optimizers,  state['optimizer_states']): o.load_state_dict(s)
        for sc, s in zip(self.schedulers, state['scheduler_states']): sc.load_state_dict(s)
        for path, s in zip(self.best_model_params_paths, state['best_model_states']):
            torch.save(s, path)
        restored_log = pickle.loads(state['log'])
        restored_log.on_epoch_end = self.log.on_epoch_end
        self.log.__dict__.update(restored_log.__dict__)
        print(f'Resumed from checkpoint — starting at epoch {self.resume_epoch}')
        print(f'Hyperparams at checkpoint: {state["hyperparams"]}')


def full_training(models : list[nn.Module],
        train_dataset : Dataset, val_dataset : Dataset, per_channel_stds : np.array,
        generator : torch.Generator,
        optimizers : list[torch.optim.Optimizer],
        schedulers : list[LRScheduler], log : LossLogger,
        batch_size : int=128, n_epochs : int=20,
        kl_weight : float=1, kl_warmup : bool=True,
        checkpoint_dir : str=None):

    with TemporaryDirectory() as tempdir:
        best_model_params_paths = [
            os.path.join(tempdir, f"best_model_params_{i}.pt")
            for i in range(len(models))
        ]

        ckpt = TrainingCheckpoint(
            models, optimizers, schedulers, log, best_model_params_paths,
            hyperparams=dict(batch_size=batch_size, n_epochs=n_epochs,
                             kl_weight=kl_weight, kl_warmup=kl_warmup))

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint.pt')):
                ckpt.load(checkpoint_dir, generator)

        for epoch in range(ckpt.resume_epoch, n_epochs + 1):
            print(f'\033[33m===  Starting epoch {epoch} of {n_epochs}... ===\033[0m')
            train_one_epoch(
                models, train_dataset, generator, optimizers, schedulers, batch_size, log,
                    kl_weight=kl_weight * min(epoch / 5, 1) if kl_warmup else kl_weight)

            print('Evaluating models on validation set...')
            per_channel_mses = []
            for modelid, (model, scheduler, best_path) in enumerate(zip(pb(models), schedulers, best_model_params_paths)):
                rlosses, kllosses, _, channel_losses = evaluate(model, val_dataset, generator, kl_weight,
                    detailed=True, subset=range(0, len(val_dataset), max(1, len(val_dataset)//2000)))
                per_channel_mses.append(channel_losses)
                scheduler.step()
                log.log_epoch(modelid, rlosses + kllosses, rlosses, kllosses, models, val_dataset)

                total_loss = rlosses.mean() + kllosses.mean()
                if total_loss < ckpt.best_val_losses[modelid]:
                    ckpt.best_val_losses[modelid] = total_loss
                    ckpt.best_epoch[modelid] = epoch
                    torch.save(model.state_dict(), best_path)

            per_channel_mses = np.stack(per_channel_mses).mean(axis=0)
            fmt = lambda a,b: ' '.join(f'{v:.2g} ({(v/b**2)*100:.0f}%)' for v in zip(a,b))
            print(f'Per-channel mean-squared error (percent of variance): \033[32m{fmt(per_channel_mses, per_channel_stds)}\033[0m')
            fmt = lambda a: ' '.join(f'{v:.2g}' for v in a)
            print(f'Best val. losses so far for each model: \033[32m{fmt(ckpt.best_val_losses)}\033[0m')
            fmt = lambda a: ' '.join(f'{v}' for v in a)
            print(f'Best epoch so far for each model: \033[32m{fmt(ckpt.best_epoch)}\033[0m')
            print()

            if checkpoint_dir is not None:
                ckpt.save(checkpoint_dir, epoch, generator)

        for model, best_path in zip(models, best_model_params_paths):
            model.load_state_dict(torch.load(best_path, weights_only=True)) # load best model states

def train_test_split(P, generator, breakdown=[0.8,0.2]):
    P.pytorch_mode()
    P.augmentation_on()
    return torch.utils.data.random_split(P, breakdown, generator=generator)

def train(models, P, kl_weight=1e-5, kl_warmup=True,
          batch_size=256, n_epochs=20, lr=1e-3, gamma=0.9,
          plot_reconstructions=False, on_epoch_end=None, seed=0, deterministic=False,
          checkpoint_dir=None):
    if seed is not None:
        set_seed(seed, deterministic=deterministic)
    g = torch.Generator(device=torch.get_default_device())
    g.manual_seed(seed)

    train_dataset, val_dataset = train_test_split(P, g)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=lr) for model in models]
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma) for optimizer in optimizers]
    log = LossLogger(log_interval=20, detailed=plot_reconstructions,
                     Pmin=P.vmin, Pmax=P.vmax,
                     on_epoch_end=on_epoch_end)

    full_training(models,
                    train_dataset, val_dataset, P.stds,
                    g,
                    optimizers, schedulers, log,
                    batch_size=batch_size, n_epochs=n_epochs,
                    kl_weight=kl_weight, kl_warmup=kl_warmup,
                    checkpoint_dir=checkpoint_dir)

    return log

def fit(models, P, Pdense, n_epochs_all=10, n_epochs_dense=20, checkpoint_dir=None, **train_kwargs):
    log1 = train(models, P, n_epochs=n_epochs_all,
                 checkpoint_dir=os.path.join(checkpoint_dir, 'phase_1') if checkpoint_dir else None,
                 **train_kwargs)
    log2 = train(models, Pdense, n_epochs=n_epochs_dense,
                 checkpoint_dir=os.path.join(checkpoint_dir, 'phase_2') if checkpoint_dir else None,
                 **train_kwargs)
    return log1, log2
