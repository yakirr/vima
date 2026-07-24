from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import pandas as pd
import random
import logging
import torch
from . import samples as vds
from .._settings import settings, logger

class ToTorch:
    """Transform converting an array to a channels-first torch tensor."""
    def __call__(self, x):
        return torch.tensor(x).permute(*range(x.ndim - 3), x.ndim-1, x.ndim-3, x.ndim-2)

class RandomDiscreteRotation:
    """Random 0/90/180/270 degree rotation augmentation."""
    def __call__(self, x):
        ntimes = np.random.choice([0,1,2,3])
        for i in range(ntimes):
            x = torch.rot90(x, dims=[-2,-1])
        return x

class PatchCollection(Dataset):
    """
    A torch Dataset of square spatial patches tiled from rasterized samples.

    Patches are laid out on a regular grid (spacing `patchstride`) over each
    sample and kept only where enough pixels are non-empty. Serves both as the
    training dataset and as the source of patch metadata (``.meta``) carried
    through the rest of the pipeline.

    Parameters
    ----------
    samples
        Maps sample ID to a ``(y, x, marker)`` DataArray, as from `read_samples`.
    patchsize
        Patch side length in pixels.
    patchstride
        Spacing in pixels between adjacent patch origins. Values below
        `patchsize` produce overlapping patches.
    max_frac_empty
        Maximum fraction of empty pixels a patch may contain and still be kept.
    normalization
        Per-marker scaling applied to patches: 'standardize' (center and scale
        to unit variance), 'center', or 'none'/None.
    percentile_thresh
        Percentile of absolute values used to set display bounds (`vmin`/`vmax`).
    covariates
        Extra per-sample covariates to condition the model on, as
        ``{name: {sid: value}}``.
    condition_on_sid
        Whether to condition the model on sample ID (default True).
    """

    @staticmethod
    def choose_patches(samples, patchsize, patchstride, max_frac_empty):
        """
        Pick patch grid positions for each sample, keeping only patches with
        enough non-empty pixels.

        Slides a `patchsize` window at spacing `patchstride` over each sample and
        retains a position only if its fraction of non-empty pixels exceeds
        ``1 - max_frac_empty``.

        Returns
        -------
        DataFrame
            Patch metadata with one row per kept patch, holding its sample ID,
            pixel origin ``(x, y)``, micron origin, and patch size.
        """
        patchmeta = []

        for s in settings.progress(samples.values(), name='choose patches'):
            mask = vds.get_mask(s)
            starts = np.array([
                [i, j]
                for i in range(0, mask.sizes['x']-patchsize, patchstride)
                for j in range(0, mask.sizes['y']-patchsize, patchstride)
                if mask.data[j:j+patchsize, i:i+patchsize].mean() > (1-max_frac_empty)
            ]).astype('int')

            patchmeta.append(pd.DataFrame([
                    (s.sid, i, j, mask.x[i], mask.y[j])
                    for i, j in starts
                ],
                columns=['sid','x','y', 'x_microns', 'y_microns'],
            ))
        patchmeta = pd.concat(patchmeta, axis=0).reset_index(drop=True)
        patchmeta.x = patchmeta.x.astype('int')
        patchmeta.y = patchmeta.y.astype('int')
        patchmeta.x_microns = patchmeta.x_microns.astype('float32')
        patchmeta.y_microns = patchmeta.y_microns.astype('float32')
        patchmeta['patchsize'] = patchsize
        return patchmeta
    
    def make_patchmeta(self, covariates=None, condition_on_sid=True):
        """
        Extract each patch's pixel array and build integer-coded covariate columns.

        Populates ``self.patches`` from the chosen patch positions and adds one
        factorized column per conditioning covariate, including sample ID (when
        `condition_on_sid`) and any extra `covariates`.

        Parameters
        ----------
        covariates
            Extra per-sample covariates as ``{name: {sid: value}}``.
        condition_on_sid
            Whether to add sample ID as a conditioning covariate.
        """
        self.patches = np.array([
            self.samples[s].data[y:y+ps,x:x+ps,:]
            for s, x, y, ps in self.meta[['sid','x','y','patchsize']].values
            ])
        self.meta['sid_num'] = pd.factorize(self.meta.sid)[0]
        if condition_on_sid:
            self._covariate_cols.append('sid_num')
        if covariates:
            for name, mapping in covariates.items():
                col = f'{name}_num'
                self.meta[col] = pd.factorize(self.meta.sid.map(mapping))[0]
                self._covariate_cols.append(col)

    def compute_stats(self, percentile_thresh):
        """
        Compute per-marker mean, std, and display percentiles over a random
        subset of patches.

        Stores ``means``/``stds`` and the display bounds ``vmin``/``vmax`` derived
        from the `percentile_thresh` percentile of absolute values.
        """
        ix = np.random.choice(len(self), min(50000, len(self)), replace=False)
        subset = self.patches[ix]
        self.means = subset.mean(axis=(0,1,2), dtype=np.float64).astype(np.float32)
        self.stds = subset.std(axis=(0,1,2), dtype=np.float64).astype(np.float32)
        self.percentiles = np.percentile(np.abs(subset), percentile_thresh, axis=(0,1,2))
        self.vmin = (-self.means - self.percentiles)/self.stds
        self.vmax = (-self.means + self.percentiles)/self.stds

        if logger.isEnabledFor(logging.DEBUG):
            fmt = lambda a: '  '.join(f'{v:.2g}' for v in a)
            logger.debug(f'per-channel means: {fmt(self.means)}')
            logger.debug(f'per-channel stds:  {fmt(self.stds)}')

    def normalize(self, normalization):
        """
        Apply the chosen per-marker normalization to the stored patches.

        Parameters
        ----------
        normalization
            'standardize' (center and scale to unit variance), 'center', or
            'none'/None.
        """
        if normalization is not None and normalization not in ['center', 'standardize', 'none']:
            raise ValueError('normalization must equal "standardize" | "center" | "none" | None')
        
        logger.debug(f'Normalizing color channels (normalization={normalization})...')

        self.empty = np.zeros(self.patches.shape[-1], dtype=np.float32)
        if normalization == 'standardize' or normalization == 'center':
            self.patches = self.patches - self.means[None,None,None,:]
            self.empty -= self.means
        if normalization == 'standardize':
            self.patches = self.patches / self.stds[None,None,None,:]
            self.empty /= self.stds
        self.normalization = normalization

    def __init__(self, samples, patchsize=40, patchstride=10, max_frac_empty=0.8,
                normalization='standardize', percentile_thresh=99,
                covariates=None, condition_on_sid=True):
        self.samples = samples
        self.patchstride = patchstride
        self._covariate_cols = []
        self.meta = PatchCollection.choose_patches(samples, patchsize, patchstride, max_frac_empty)
        self.nmarkers = next(iter(samples.values())).sizes['marker']

        self.pytorch_mode()
        self.make_patchmeta(covariates=covariates, condition_on_sid=condition_on_sid)
        self.compute_stats(percentile_thresh)
        self.normalize(normalization=normalization)
        self.augmentation_off()

    def refined(self, max_frac_empty, tol=1e-10, normalization='standardize', percentile_thresh=99):
        """
        Return a copy restricted to denser patches.

        Keeps patches whose fraction of empty pixels is below `max_frac_empty`
        and recomputes normalization statistics on the retained subset. Used to
        refine training on tissue-rich patches after an initial pass.

        Parameters
        ----------
        max_frac_empty
            Maximum fraction of empty pixels a patch may contain and still be kept.
        """
        import copy
        empty_frac = (np.abs(self.patches - self.empty[None,None,None,:]).max(axis=-1) < tol).mean(axis=(1, 2))
        keep = np.where(empty_frac < max_frac_empty)[0]
        result = copy.copy(self)
        result.subset(keep,
                      normalization=normalization, percentile_thresh=percentile_thresh)
        return result

    @property
    def sid_nums(self):
        """Map each sample ID to its integer code."""
        return {sid:sid_num for sid, sid_num in self.meta[['sid','sid_num']].drop_duplicates().values}

    @property
    def nsamples(self):
        """Number of distinct samples represented across the patches."""
        return len(self.meta.sid.unique())

    @property
    def covariate_sizes(self):
        """Number of categories in each conditioning covariate."""
        return [self.meta[col].nunique() for col in self._covariate_cols]

    def augmentation_on(self):
        """Enable random rotation and horizontal flip augmentation (pytorch mode only)."""
        if self.dim_order != 'pytorch':
            logger.warning('Data augmentation only available in pytorch mode. Will leave augmentation off')
            return
        logger.debug('[PatchCollection: data augmentation on]')
        self.transform = transforms.Compose([
            ToTorch(),
            RandomDiscreteRotation(),
            transforms.RandomHorizontalFlip(),
            ])
    def augmentation_off(self):
        """Disable rotation and flip augmentation."""
        logger.debug('[PatchCollection: data augmentation is off]')
        self.transform = transforms.Compose([
            ToTorch(),
            ])

    def add_donor_ids(self, donor_ids_series):
        """Map a donor ID onto each patch via its sample ID."""
        self.meta['donor'] = self.meta.sid.map(donor_ids_series)

    def pytorch_mode(self):
        """Switch patch output to channels-first (C, H, W) torch layout."""
        self.dim_order = 'pytorch'
        logger.debug('[PatchCollection: in pytorch mode]')
    def numpy_mode(self):
        """Switch patch output to ``(H, W, C)`` numpy layout and turn augmentation off."""
        self.dim_order = 'numpy'
        self.augmentation_off()
        logger.debug('[PatchCollection: in numpy mode]')

    def subset(self, ix, percentile_thresh, normalization):
        """
        Restrict the collection in place to the given patch indices and recompute
        normalization statistics.
        """
        self.patches = self.patches[ix]
        self.meta = self.meta.iloc[ix]
        self.compute_stats(percentile_thresh)
        self.normalize(normalization=normalization)

    def __repr__(self):
        ps = self.meta.patchsize.iloc[0]
        cov_parts = [f'{col.removesuffix("_num")} ({self.meta[col].nunique()} values)'
                     for col in self._covariate_cols]
        cov_str = ', '.join(cov_parts) if cov_parts else 'none'
        return (
            f'PatchCollection object with npatches × width × height × nmarkers = {len(self)}×{ps}×{ps}×{self.nmarkers}\n'
            f'\tcovariates: {cov_str}'
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        patches = self.patches[idx]
        covars = self.meta[self._covariate_cols].values[idx]
        if self.dim_order == 'numpy':
            return patches, covars
        else:
            return self.transform(patches), torch.tensor(covars, dtype=torch.long)