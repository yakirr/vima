import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import torch
from torch.utils.data import DataLoader
import cna
import warnings
import scipy.sparse as sp
import scipy.stats as st
from argparse import Namespace
from tqdm import tqdm
from .fingerprints import Fingerprints
pb = lambda x: tqdm(x, ncols=100)

def anndata(patchmeta, Z, var_names=None, use_rep='X', n_comps=10, **kwargs):
    d = ad.AnnData(Z)
    if var_names is not None:
        d.var_names = var_names
    obs = patchmeta.copy()
    obs.index = obs.index.astype(str)
    d.obs = obs

    if use_rep == 'X_pca':
        sc.tl.pca(d, n_comps=min(n_comps, Z.shape[1]-1))

    sc.pp.neighbors(d, use_rep=use_rep, **kwargs)

    return d

def apply(models, P, batch_size=1000, with_mse=False):
    P.pytorch_mode()
    P.augmentation_off()
    for model in models:
        model.eval()

    eval_loader = DataLoader(
        dataset=P,
        batch_size=batch_size,
        shuffle=False)

    Zs = {modelid: [] for modelid in range(len(models))}
    if with_mse:
        MSEs = {modelid: [] for modelid in range(len(models))}
    with torch.no_grad():
        for batch in pb(eval_loader):
            for modelid, model in enumerate(models):
                if with_mse:
                    x_recon, mean, _ = model(batch, sample_from_latent=False)
                    Zs[modelid].append(mean.reshape(len(batch[0]), -1).detach().cpu().numpy())
                    MSEs[modelid].append(((x_recon - batch[0]) ** 2).mean(dim=(2, 3)).detach().cpu().numpy())
                else:
                    Zs[modelid].append(model.embedding(batch).detach().cpu().numpy())

    Zs_out = np.array([np.concatenate(Z) for Z in Zs.values()])
    if with_mse:
        MSEs_out = np.array([np.concatenate(M) for M in MSEs.values()])
        return Zs_out, MSEs_out
    return Zs_out

def latentreps(models, P, use_rep='X', n_comps=100, with_mse=True, **kwargs):
    print('applying models')
    result = apply(models, P, with_mse=with_mse)
    Zs, MSEs = result if with_mse else (result, None)

    print('computing nearest-neighbor graphs')
    ds = [anndata(P.meta, Z, use_rep=use_rep, n_comps=n_comps, **kwargs) for Z in pb(Zs)]
    fp = Fingerprints.from_list(ds)
    
    if MSEs is not None:
        mean_mse = MSEs.mean(axis=0)
        marker_names = next(iter(P.samples.values())).coords['marker'].values
        per_channel = pd.DataFrame(mean_mse, index=fp.obs.index, columns=marker_names)
        fp.obsm['per_channel_mse'] = per_channel
        fp.obs['mse'] = per_channel.mean(axis=1)
    return fp 


def tail_counts(z, znull):
    if znull.ndim == 1:
        znull = znull[:, None]

    z2 = np.square(z)
    zn2 = np.sort(np.square(znull), axis=0)

    return np.array([
        zn2.shape[0] - np.searchsorted(zn2[:, j], z2, side="left")
        for j in range(zn2.shape[1])
    ])

def empirical_fdrs(z, znull, thresholds):
    if znull.shape[0] != len(z):
        raise ValueError("shape mismatch")

    tails = tail_counts(thresholds, znull)
    ranks = len(z) - np.searchsorted(np.sort(z**2), thresholds**2, side="left")

    return (tails / ranks).mean(axis=0)


def _power_ratio(x, power, axis):
    """Meta-analysis power ratio (x**power).sum / (x**2).sum along `axis` (the model axis)."""
    return (x**power).sum(axis=axis) / (x**2).sum(axis=axis)


def _association(MAMresid, M, y, batches, donorids, rng, Nnull=10_000,
                 max_num_mns=1_000, show_progress=False):
    # prep data
    y = (y - y.mean())/y.std()
    n = len(y)
    ycond = M.dot(y)
    ycond /= ycond.std(axis=0)

    # make null phenotypes
    if donorids is not None:
        y_ = cna.tl._stats.grouplevel_permutation(donorids, y, Nnull)
    else:
        y_ = cna.tl._stats.conditional_permutation(batches, y, Nnull)
    ycond_ = M.dot(y_)
    ycond_ /= ycond_.std(axis=0)

    # get microniche coefficients and weights (over all patches)
    mncorrs = (ycond[:,None,None]*MAMresid).mean(axis=0)
    weights = (mncorrs**2) / (mncorrs**2).sum(axis=0)
    mncorrs_meta = _power_ratio(mncorrs, 3, axis=0)

    # subsample patches (last axis of MAMresid) for the expensive global/FDR machinery
    Npatches = MAMresid.shape[2]
    if Npatches > max_num_mns:
        sub = rng.choice(Npatches, size=max_num_mns, replace=False)
    else:
        sub = np.arange(Npatches)
    MAMresid_sub = MAMresid[:, :, sub]
    mncorrs_sub = mncorrs[:, sub]
    mncorrs_meta_sub = mncorrs_meta[sub]

    # meta-analyzed mn coefficients and global test statistics (on the subsampled patches)
    nullmncorrs = np.einsum('nk,nmp->kmp', ycond_, MAMresid_sub) / n
    globalstat = _power_ratio(mncorrs_sub, 4, axis=0).mean()
    nullglobalstats = _power_ratio(nullmncorrs, 4, axis=1).mean(axis=1)
    nullmncorrs_meta = _power_ratio(nullmncorrs, 3, axis=1).T

    # compute global p-vaule
    p = ((nullglobalstats >= globalstat).sum() + 1)/(len(nullglobalstats) + 1)
    print(f'\033[32mP = {p}\033[0m')
    if p <= 1/(Nnull + 1)+1e-10:
        warnings.warn('global association p-value attained minimal possible value. '+\
                'Consider increasing Nnull')

    thr = np.quantile(np.abs(mncorrs_meta_sub), np.arange(0.01, 1, 0.01))
    fdrs = empirical_fdrs(mncorrs_meta_sub, nullmncorrs_meta, thr)
    fdrs = pd.DataFrame({
        'threshold':thr,
        'fdr':fdrs})

    res = {'p':p, 'mncorrs':mncorrs_meta, 'fdrs':fdrs,
            'globalstat':globalstat, 'nullglobalstats':nullglobalstats,
            'weights':weights,
            'nullmncorrs':nullmncorrs_meta,
            'permodel_mncorrs':mncorrs,
            'MAMres':MAMresid,
            'ycond':ycond
            }

    return Namespace(**res)


def compute_mams(ds, sid_name, nsteps=None, self_weight=1, show_progress=False):
    """Compute the per-model MAT (list of raw NAM DataFrames) once so it can be reused across
    phenotypes via association(..., MAMs=...). This performs only the (expensive) diffusion;
    batch QC and sample/covariate filtering are applied later inside association(), so no batch
    information is needed here and a single precomputed MAT can be reused across different
    batch/covariate/phenotype specifications."""
    print('computing MAT') #TODO: rename MAM to MAT in code if we keep this nomenclature
    MAMs = []
    for d in tqdm(ds.modelspecific_fingerprints(), total=ds.nmodels, ncols=100):
        NAM = cna.tl._nam._nam(d, sid_name, nsteps=nsteps, self_weight=self_weight,
                               show_progress=show_progress)
        MAMs.append(NAM)
    return MAMs

def association(ds, y, sid_name, batches=None, covs=None, donorids=None, key_added='mncoef',
                return_full=False, ridges=None, MAMs=None,
                Nnull=10_000, seed=0, make_umap=True,
                nsteps=None, show_progress=False, allow_low_sample_size=False,
                max_num_mns=5_000, **kwargs):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # Check formats of inputs and figure out which samples have valid data
    batches, filter_samples = cna.tl._association.check_inputs(ds.select_model(0), y, sid_name, batches, covs, donorids, allow_low_sample_size)

    # Compute raw NAMs (unless precomputed), then apply batch QC and sample/column filtering
    if MAMs is None:
        MAMs = compute_mams(ds, sid_name, nsteps=nsteps, show_progress=show_progress)
    elif len(MAMs) != ds.nmodels:
        raise ValueError(f'Expected MAMs of length {ds.nmodels}, got {len(MAMs)}.')

    MAMs_filtered = []
    kepts = []
    for NAM in MAMs:
        NAMqc, keep = cna.tl._nam._qc_nam(NAM, batches, show_progress=show_progress)
        NAM, kept, batches, covs, donorids, filter_samples = cna.tl._association.reindex_and_filter_nam(
            NAMqc, keep, y, batches, covs, donorids, filter_samples)
        MAMs_filtered.append(NAM)
        kepts.append(kept)
    kept = np.logical_and.reduce(kepts)

    for i in range(len(MAMs_filtered)):
        MAMs_filtered[i] = MAMs_filtered[i][ds.obs.index[kept]]

    # residualize NAMs
    MAMs_concat = pd.concat(MAMs_filtered, axis=1)
    MAMs_concat.columns = range(MAMs_concat.shape[1])
    res = cna.tl._nam._resid_nam(MAMs_concat,
                        covs[filter_samples] if covs is not None else covs,
                        batches[filter_samples] if batches is not None else batches,
                        npcs=1,
                        ridges=ridges,
                        show_progress=show_progress)
    MAMs_concat = res.namresid

    print('performing association test')
    n_samples, n_total = MAMs_concat.shape
    Npatches = n_total // ds.nmodels
    MAMresid = MAMs_concat.values.reshape(n_samples, ds.nmodels, Npatches)
    res_ = _association(
        MAMresid, res.M.values,
        y[filter_samples].values, batches[filter_samples].values,
        donorids[filter_samples].values if donorids is not None else None,
        rng,
        max_num_mns=max_num_mns,
        show_progress=show_progress, Nnull=Nnull,
        **kwargs)
    res.__dict__.update(vars(res_)) # add info from from res_ to res
    res.kept = kept
    
    # make anndata with results
    D = ds.weighted_avg_graph(res.weights, kept, make_umap=make_umap)
    if key_added in D.obs:
        warnings.warn(f"Key '{key_added}' already exists in d.obs. Overwriting.")
    D.obs[key_added] = res.mncorrs
    D.obsm['permodel_mncorrs'] = pd.DataFrame(res.permodel_mncorrs.T,
                                              columns=[f'model{i}' for i in range(1, ds.nmodels+1)],
                                              index=D.obs.index)
    
    # compute local FDRs (vectorized: min fdr over all thresholds <= |mncorr|)
    thr_sorted = res.fdrs.threshold.values          # ascending, from np.quantile
    cummin_fdr = np.minimum.accumulate(res.fdrs.fdr.values)
    k = np.searchsorted(thr_sorted, np.abs(D.obs[key_added].values), side='right')
    D.obs[f'{key_added}_fdr'] = np.where(k > 0, cummin_fdr[np.clip(k - 1, 0, None)], 1.0)
    D.uns['vima_p'] = res.p
    D.uns['vima_pheno'] = y.name

    if return_full:
        return res, D
    else:
        return res.p, D