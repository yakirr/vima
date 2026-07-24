import os
from concurrent.futures import ThreadPoolExecutor
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
from .fingerprints import Fingerprints
from ._settings import Verbosity, settings, logger

def anndata(patchmeta, Z, var_names=None, use_rep='X', n_comps=10, **kwargs):
    """
    Build an AnnData with a neighbor graph from an embedding matrix.

    Parameters
    ----------
    patchmeta
        Per-patch metadata to store in ``.obs``.
    Z
        Patch-by-dimension embedding matrix.
    use_rep
        Representation used to build the neighbor graph: 'X' (the embedding) or
        'X_pca'.
    n_comps
        Number of PCs when ``use_rep='X_pca'``.

    Returns
    -------
    AnnData
        Embedding in ``X``, patch metadata in ``.obs``, and a nearest-neighbor
        graph.
    """
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
    """
    Run a trained model ensemble over a PatchCollection to embed every patch.

    Runs with augmentation disabled and each model in eval mode.

    Parameters
    ----------
    models
        Trained ensemble from `cVAE`.
    P
        Patches to embed.
    with_mse
        Also return per-patch, per-channel reconstruction error.

    Returns
    -------
    ndarray or tuple
        Model-by-patch-by-dimension embeddings; if `with_mse`, also a
        model-by-patch-by-marker array of reconstruction MSEs.
    """
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
        for batch in settings.progress(eval_loader, name='apply models'):
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
    """
    Compute patch fingerprints from a trained model ensemble.

    Applies each model to every patch to obtain its latent embedding, then
    builds a per-model nearest-neighbor graph over the patches. These are the
    "fingerprints" consumed by `association`.

    Parameters
    ----------
    models
        Trained ensemble from `cVAE`.
    P
        Patches to embed (typically the refined collection).
    use_rep
        Representation used to build the neighbor graph: 'X' (the embedding) or
        'X_pca'.
    n_comps
        Number of PCs when ``use_rep='X_pca'``.
    with_mse
        Also store per-patch and per-channel reconstruction error.

    Returns
    -------
    Fingerprints
        Per-model embeddings and neighbor graphs, with ``.obs`` carrying patch
        metadata.
    """
    logger.info('applying models')
    result = apply(models, P, with_mse=with_mse)
    Zs, MSEs = result if with_mse else (result, None)

    logger.info('computing nearest-neighbor graphs')
    ds = [anndata(P.meta, Z, use_rep=use_rep, n_comps=n_comps, **kwargs) for Z in settings.progress(Zs, name='nearest-neighbor graphs')]
    fp = Fingerprints.from_list(ds)
    
    if MSEs is not None:
        mean_mse = MSEs.mean(axis=0)
        marker_names = next(iter(P.samples.values())).coords['marker'].values
        per_channel = pd.DataFrame(mean_mse, index=fp.obs.index, columns=marker_names)
        fp.obsm['per_channel_mse'] = per_channel
        fp.obs['mse'] = per_channel.mean(axis=1)
    return fp 


def _tail_counts_total(znull, t2_sorted, nthreads):
    """Total count of znull**2 entries >= each (sorted) squared threshold.
    """
    col_chunks = np.array_split(np.arange(znull.shape[1]), nthreads)

    def chunk_hist(cols):
        sub = np.square(znull[:, cols]).ravel()
        # number of thresholds each entry exceeds, in [0, len(t2_sorted)]
        m = np.searchsorted(t2_sorted, sub, side="right")
        return np.bincount(m, minlength=t2_sorted.size + 1)

    with ThreadPoolExecutor(nthreads) as ex:
        hist = np.sum(list(ex.map(chunk_hist, col_chunks)), axis=0)

    # counts_sorted[i] = # entries exceeding at least (i+1) thresholds
    return np.cumsum(hist[::-1])[::-1][1:]


def empirical_fdrs(z, znull, thresholds):
    """
    Compute the empirical FDR at each threshold from observed and null statistics.

    At each threshold the FDR estimate is the mean number of null statistics
    exceeding it (per null realization) divided by the number of observed
    statistics exceeding it, comparing magnitudes.

    Parameters
    ----------
    z
        Observed per-microniche statistics.
    znull
        Null statistics, one column per permutation (aligned with `z`).
    thresholds
        Thresholds at which to evaluate the FDR.

    Returns
    -------
    ndarray
        Estimated FDR at each threshold.
    """
    if znull.shape[0] != len(z):
        raise ValueError("shape mismatch")

    if znull.ndim == 1:
        znull = znull[:, None]
    ncols = znull.shape[1]

    t2 = np.square(np.asarray(thresholds, dtype=float))
    order = np.argsort(t2)

    nthreads = min(ncols, (os.cpu_count() or 1))
    counts_sorted = _tail_counts_total(znull, t2[order], nthreads)
    mean_tails = np.empty(t2.shape)
    mean_tails[order] = counts_sorted / ncols

    ranks = len(z) - np.searchsorted(np.sort(np.square(z)), t2, side="left")

    return mean_tails / ranks


def _power_ratio(x, power, axis):
    """Meta-analysis power ratio (x**power).sum / (x**2).sum along `axis` (the model axis)."""
    return (x**power).sum(axis=axis) / (x**2).sum(axis=axis)


def _association(MAMresid, M, y, batches, donorids, rng, Nnull=10_000,
                 max_num_mns=5_000, show_progress=False):
    """
    Run the permutation-based microniche association test.

    Correlates each microniche's residualized cross-sample abundance with the
    covariate-conditioned phenotype, meta-analyzing across models, and calibrates
    both a global p-value and per-microniche FDRs against permutation nulls. Null
    phenotypes are drawn at the donor level when `donorids` is given, or within
    `batches` if that is given, or unconditionally otherwise.

    Parameters
    ----------
    MAMresid
        Residualized abundances, shaped sample-by-model-by-microniche.
    M
        Residualization matrix (sample-by-sample) used to residualize any covariates
        out of MAMresid.
    y
        Sample-level phenotype.
    max_num_mns
        Cap on microniches subsampled for the global and FDR computations.

    Returns
    -------
    Namespace
        Global p-value, per-microniche and per-model coefficients, mixing
        weights, FDR table, and the associated null distributions.
    """
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

    # meta-analyzed mn coefficients and global test statistics (on the subsampled patches).
    # We loop over the (few) models and accumulate the per-model power sums Sq = sum_m nm_m**q
    # incrementally, so we never materialize the full (Nnull, Nmodels, max_num_mns) array.
    globalstat = _power_ratio(mncorrs_sub, 4, axis=0).mean()
    ycond_T = np.ascontiguousarray(ycond_.T, dtype=np.float32)      # (Nnull, n)
    MAMresid_sub32 = MAMresid_sub.astype(np.float32)
    S2 = np.zeros((ycond_.shape[1], MAMresid_sub.shape[2]), dtype=np.float32)  # (Nnull, max_num_mns)
    S3 = np.zeros_like(S2); S4 = np.zeros_like(S2)
    MAMresid_sub32_permodel = [np.ascontiguousarray(MAMresid_sub32[:, m, :]) for m in range(MAMresid_sub32.shape[1])]
    for myMAM in MAMresid_sub32_permodel:
        nm = (ycond_T @ myMAM) / n            # null mn coeffs for model m: (Nnull, npatch)
        nm2 = nm * nm
        S2 += nm2; S3 += nm2 * nm; S4 += nm2 * nm2
    
    nullglobalstats = (S4 / S2).mean(axis=1)
    nullmncorrs_meta = (S3 / S2).T

    # compute global p-vaule
    p = ((nullglobalstats >= globalstat).sum() + 1)/(len(nullglobalstats) + 1)
    settings.result(f'\033[32mP = {p}\033[0m')
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


def compute_mams(ds, sid_name, nsteps=None, self_weight=1, show_progress=None):
    """
    Precompute per-model sample-by-microniche abundance matrices.

    Runs only the expensive graph-diffusion step, independent of any phenotype,
    batch, or covariate choice. Pass the result to ``association(..., MAMs=...)``
    to reuse it across multiple phenotypes without recomputing.

    Parameters
    ----------
    ds
        Fingerprints from `latentreps`.
    sid_name
        Column in ``ds.obs`` giving each patch's sample ID.

    Returns
    -------
    list
        One abundance matrix per model.
    """
    if show_progress is None:
        show_progress = settings.verbosity >= Verbosity.verbose
    logger.info('computing MAT') #TODO: rename MAM to MAT in code if we keep this nomenclature
    MAMs = []
    for d in settings.progress(ds.modelspecific_fingerprints(), total=ds.nmodels, name='compute MAMs'):
        NAM = cna.tl._nam._nam(d, sid_name, nsteps=nsteps, self_weight=self_weight,
                               show_progress=show_progress)
        MAMs.append(NAM)
    return MAMs

def association(ds, y, sid_name, batches=None, covs=None, donorids=None, key_added='mncoef',
                return_full=False, ridges=None, MAMs=None,
                Nnull=10_000, seed=0, make_umap=True,
                nsteps=None, show_progress=None, allow_low_sample_size=False,
                max_num_mns=5_000, **kwargs):
    """
    Test patch fingerprints for association with a sample-level phenotype.

    Fits a covariate-aware model relating each microniche's cross-sample
    abundance to the phenotype, assessing significance by permutation. Returns
    both a global p-value for the dataset and a per-microniche coefficient with
    an empirical FDR.

    Parameters
    ----------
    ds
        Fingerprints from `latentreps`.
    y
        Sample-level phenotype indexed by sample ID (e.g. case/control).
    sid_name
        Column in ``ds.obs`` giving each patch's sample ID.
    batches
        Sample-level batch labels to condition on.
    covs
        Sample-level covariates to control for.
    donorids
        Sample-level donor IDs; when given, permutations are done at the donor
        level to respect repeated samples per donor. This cannot be used together
        with `batches`.
    key_added
        Name for the per-microniche coefficient column written to ``D.obs``.
    return_full
        If True, return the full internal result object instead of just the
        p-value (see Returns).
    MAMs
        Precomputed abundance matrices from `compute_mams`; recomputed if None.
    Nnull
        Number of null permutations.
    make_umap
        Compute a UMAP of the microniche graph for visualization.
    max_num_mns
        Cap on the number of microniches used for the statistical tests.

    Returns
    -------
    tuple
        ``(p, D)`` where `p` is the global association p-value and `D` is an
        AnnData of microniches carrying per-microniche coefficients
        (``D.obs[key_added]``) and FDRs (``D.obs[f'{key_added}_fdr']``). If
        `return_full` is True, returns ``(res, D)`` with the full result object
        in place of `p`.
    """
    if show_progress is None:
        show_progress = settings.verbosity >= Verbosity.verbose
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

    logger.info('performing association test')
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