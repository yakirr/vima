import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from IPython import display
from .. import vis as v

class LossLogger:
    def __init__(self, log_interval=20, detailed=True, Pmin=None, Pmax=None, on_epoch_end=None):
        self.log_interval = log_interval
        self.detailed = detailed
        self.Pmin = Pmin; self.Pmax = Pmax
        self.on_epoch_end = on_epoch_end
        self.reset()

    def reset(self):
        """Reset all logged values."""
        self.chunkstarttime = time.time()
        self.epochstarttime = time.time()
        self.starttime = time.time()
        self.epoch = 1

        self.epochs = []
        self.batch_nums = []
        self.losses = []
        self.rlosses = []
        self.kllosses = []
        self.klweights = []
        self.lrs = []
        self.modelids = []

        self.val_losses = []
        self.val_rlosses = []
        self.val_kllosses = []

    @property
    def nmodels(self):
        """Return the number of models being trained."""
        return len(set(self.modelids))

    def log_batch(self, modelid, batch_num, loss, rloss, klloss, klweight, lr):
        """Log losses for a single batch."""
        self.epochs.append(self.epoch)
        self.batch_nums.append(batch_num)
        self.modelids.append(modelid)
        self.losses.append(float(loss.detach()))
        self.rlosses.append(float(rloss.detach()))
        self.kllosses.append(float(klloss.detach()))
        self.klweights.append(float(klweight))
        self.lrs.append(float(lr))
        self.val_losses.append(np.nan)
        self.val_rlosses.append(np.nan)
        self.val_kllosses.append(np.nan)

        if batch_num % self.log_interval == 0 and batch_num > 0 and modelid == 0:
            self.print_batch_log(batch_num, klweight, lr)
    
    def print_batch_log(self, batch_num, kl_weight, lr):
        """Print batch-level logging information."""
        time_per_batch = (time.time() - self.chunkstarttime) / self.log_interval
        self.chunkstarttime = time.time()

        cur_rloss = np.mean(self.rlosses[-self.log_interval*self.nmodels:])
        cur_klloss = np.mean(self.kllosses[-self.log_interval*self.nmodels:])

        print(f'\033[90mbatch {batch_num:5d} | '
                f'lr {lr:.2g} | '
                f'r-loss {cur_rloss:.2f} | '
                f'kl-loss {cur_klloss:.2f} | '
                f'kl-weight {kl_weight:.2e} | '
                f'{time_per_batch:.2f} sec/batch\033[0m')

    def log_epoch_onemodel(self, modelid, vlosses, vrlosses, vkllosses):
        """Log validation losses for one model."""
        i = np.where(np.array(self.modelids) == modelid)[0][-1]
        self.val_losses[i] = vlosses.mean()
        self.val_rlosses[i] = vrlosses.mean()
        self.val_kllosses[i] = vkllosses.mean()

    def finalize_epoch(self, vrlosses, models, val_dataset,
                       per_channel_mses, per_channel_stds, best_val_losses, best_epochs):
        """Call once after all models are evaluated. Prints, plots, fires callback, advances counter."""
        self._print_epoch_log(vrlosses, models, val_dataset)
        fmt_ch = lambda a, b: ', '.join(f'{mse:.2g} ({(mse/std**2)*100:.0f}%)' for mse, std in zip(a, b))
        fmt    = lambda a: ', '.join(f'{v:.2g}' for v in a)
        fmt_i  = lambda a: ', '.join(f'{v}' for v in a)
        print(f'Per-channel MSE (% of variance): \033[32m{fmt_ch(per_channel_mses, per_channel_stds)}\033[0m')
        print(f'Best val. losses so far: \033[32m{fmt(best_val_losses)}\033[0m')
        print(f'Best epoch so far: \033[32m{fmt_i(best_epochs)}\033[0m')
        if self.on_epoch_end is not None:
            self.on_epoch_end(self.epoch, models, val_dataset)
        self.epoch += 1
        self.epochstarttime = time.time()
        self.chunkstarttime = time.time()

    def _print_epoch_log(self, vrlosses, models, val_dataset):
        display.clear_output()
        plt.figure(figsize=(9,3))
        plt.subplot(1,2,1)
        info = self.summary.groupby(['epoch', 'batch_num'], as_index=False).mean()

        plt.plot(info.loss, label='total loss', alpha=0.5)
        for modelid in range(self.nmodels):
            toplot = self.summary[self.summary.modelid == modelid]
            plt.scatter(info.index.values, toplot.val_loss.values, marker='x', label='val loss')
        plt.scatter(np.argmin(info.val_loss), info.val_loss.min(), color='red')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.title('Train. loss (line) & per-model val. loss (Xs)')
        plt.ylim(0.8*np.min(info.loss.values), 1.1*np.percentile(info.loss.values, 95))
        plt.xlabel('Batch number')
        plt.ylabel('Loss')
        
        plt.subplot(1,2,2)
        plt.hist(vrlosses, bins=50)
        plt.title('Reconstruction error across validation patches')
        plt.xlabel('Reconstruction error'); plt.ylabel('#Patches')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.show()

        if self.detailed:
            ix = np.argsort(vrlosses)
            examples = val_dataset[list(ix[::len(ix)//12])]
            examples = (examples[0].permute(0,2,3,1), examples[1])
            v.plot_with_reconstruction(models[0], examples, channels=range(examples[0].shape[-1]),
                                        pmin=self.Pmin, pmax=self.Pmax)
            
        print(f'End of epoch {self.epoch}')
        print(f'Avg. val. loss: \033[32m{np.array(self.val_losses)[-self.nmodels:].mean()}\033[0m')
        print(f'Time elapsed this epoch: \033[34m{time.time() - self.epochstarttime:.2f} sec\033[0m')
        print(f'Total time elapsed: \033[34m{time.time() - self.starttime:.2f} sec\033[0m')

    @property
    def summary(self):
        """Return a summary of the logged losses as a DataFrame."""
        return pd.DataFrame({
            'epoch': self.epochs,
            'batch_num': self.batch_nums,
            'modelid': self.modelids,
            'lr': self.lrs,
            'klweight': self.klweights,
            'loss': self.losses,
            'rloss': self.rlosses,
            'klloss': self.kllosses,
            'val_loss': self.val_losses,
            'val_rloss': self.val_rlosses,
            'val_klloss': self.val_kllosses
        })