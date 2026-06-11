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

def omnibus_pvalue(obs, null, Nnull, mn_chunk=None):
    """obs: (Nmodels, Nmns).  null: (Nnull, Nmodels, Nmns) ndarray or memmap.
    Returns (P_value, n_nulls_ge_obs)."""
    K, _, M = null.shape
 
    # reduce models -> per-microniche (level, spread); vectorized, no null loop
    a_n = null.mean(axis=1)
    v_n = null.var(axis=1)
    a_o = obs.mean(0).astype(np.float64)
    v_o = obs.var(0).astype(np.float64)
 
    # sufficient statistics over nulls (accumulated in float64)
    Sa  = a_n.sum(0, dtype=np.float64);            Sv  = v_n.sum(0, dtype=np.float64)
    Saa = np.einsum('km,km->m', a_n, a_n, dtype=np.float64)
    Svv = np.einsum('km,km->m', v_n, v_n, dtype=np.float64)
    Sav = np.einsum('km,km->m', a_n, v_n, dtype=np.float64)
 
    def quad(dx, dy, caa, cvv, cav):
        det = np.maximum(caa * cvv - cav * cav, 1e-300)
        return (dx * dx * cvv - 2 * dx * dy * cav + dy * dy * caa) / det
 
    # obs scored against the full-null (K-sample) moments
    ma, mv = Sa / K, Sv / K
    caa = (Saa - K * ma * ma) / (K - 1)
    cvv = (Svv - K * mv * mv) / (K - 1)
    cav = (Sav - K * ma * mv) / (K - 1)
    D_obs = float(quad(a_o - ma, v_o - mv, caa, cvv, cav).sum())
 
    # all nulls at once via closed-form leave-one-out; chunk only to cap memory
    n = K - 1
    D_null = np.zeros(K)
    if mn_chunk is None:
        mn_chunk = max(1, min(M, int(2e7 // max(K, 1))))
    for s in range(0, M, mn_chunk):
        sl = slice(s, s + mn_chunk)
        a = a_n[:, sl].astype(np.float64)            # (K, b)
        v = v_n[:, sl].astype(np.float64)
        Sa_b, Sv_b = Sa[sl], Sv[sl]
        Saa_b, Svv_b, Sav_b = Saa[sl], Svv[sl], Sav[sl]
        ma = (Sa_b - a) / n
        mv = (Sv_b - v) / n
        dx = a - ma                                  # = (K*a - Sa_b)/n
        dy = v - mv
        caa = ((Saa_b - a * a) - n * ma * ma) / (n - 1)
        cvv = ((Svv_b - v * v) - n * mv * mv) / (n - 1)
        cav = ((Sav_b - a * v) - n * ma * mv) / (n - 1)
        D_null += quad(dx, dy, caa, cvv, cav).sum(axis=1)
 
    ge = int(np.sum(D_null >= D_obs))
    return (1 + ge) / (1 + K), D_obs, D_null

def _association(MAMresid, M, y, batches, donorids, Nnull=1000,
                 max_num_mns=10_000, show_progress=False):
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

    # MAMresid: (n, Nmodels, Npatches)
    # get microniche coefficients and weights
    mncorrs = (ycond[:,None,None]*MAMresid).mean(axis=0)  # (Nmodels, Npatches)
    weights = (mncorrs**2) / (mncorrs**2).sum(axis=0)

    # nullmncorrs: (Nnull, Nmodels, Npatches) — subsample patches if needed
    Nmns = MAMresid.shape[2]
    if Nmns > max_num_mns:
        patch_ix = np.random.choice(Nmns, max_num_mns, replace=False)
        MAMresid_downsampled = MAMresid[:, :, patch_ix]
    else:
        patch_ix = np.ones(Nmns).astype(bool)
        MAMresid_downsampled = MAMresid
    nullmncorrs = np.einsum('nmp,nl->lmp',
                            MAMresid_downsampled.astype(np.float32),
                            ycond_.astype(np.float32)) / n

    # computed meta-analyzed mn coefficients and global test statistics
    mncorrs_meta = (mncorrs**3).sum(axis=0)/(mncorrs**2).sum(axis=0)

    # compute global p-value
    p, globalstat, nullglobalstats = omnibus_pvalue(mncorrs[:,patch_ix], nullmncorrs, Nnull)
    print(f'\033[32mP = {p}\033[0m')
    if p <= 1/(Nnull + 1)+1e-10:
        warnings.warn('global association p-value attained minimal possible value. '+\
                'Consider increasing Nnull')
    
    # thresholds must be derived from the same (subsampled) patches passed to
    # empirical_fdrs, else the highest thresholds have zero real detections (div by zero)
    mncorrs_meta_downsampled = mncorrs_meta[patch_ix]
    maxcorr = max(np.abs(mncorrs_meta_downsampled).max(), 0.001)
    fdr_thresholds = np.arange(maxcorr/4, maxcorr, maxcorr/400)
    nullmncorrs_meta = ((nullmncorrs**3).sum(axis=1)/(nullmncorrs**2).sum(axis=1)).T

    fdr_vals = cna.tl._stats.empirical_fdrs(mncorrs_meta_downsampled, nullmncorrs_meta, fdr_thresholds)
    fdrs = pd.DataFrame({
        'threshold':fdr_thresholds,
        'fdr':fdr_vals})

    res = {'p':p, 'mncorrs':mncorrs_meta, 'fdrs':fdrs,
            'globalstat':globalstat, 'nullglobalstats':nullglobalstats,
            'weights':weights,
            'nullmncorrs':nullmncorrs_meta,
            }
    return Namespace(**res), mncorrs, nullmncorrs, patch_ix


# def _association(MAMresid, M, y, batches, donorids, patches_to_samples, ks=None, Nnull=1000,
#                  max_num_mns=10_000, show_progress=False):
#     # prep data
#     y = (y - y.mean())/y.std()
#     n = len(y)
#     ycond = M.dot(y)
#     ycond /= ycond.std(axis=0)

#     # make null phenotypes
#     if donorids is not None:
#         y_ = cna.tl._stats.grouplevel_permutation(donorids, y, Nnull)
#     else:
#         y_ = cna.tl._stats.conditional_permutation(batches, y, Nnull)
#     ycond_ = M.dot(y_)
#     ycond_ /= ycond_.std(axis=0)

#     # MAMresid: (n, Nmodels, Npatches)
#     # (1) per-model microniche correlations, (2) Fisher-transformed elementwise
#     mncorrs = (ycond[:,None,None]*MAMresid).mean(axis=0)  # (Nmodels, Npatches)
#     mncorrs = np.arctanh(mncorrs)
#     weights = (mncorrs**2) / (mncorrs**2).sum(axis=0)

#     # (3) meta-analyzed correlation per patch = average over models
#     mncorrs_meta = mncorrs.mean(axis=0)  # (Npatches,)

#     # (4) nullmncorrs: (Nnull, Nmodels, Npatches) — subsample patches if needed,
#     #     Fisher-transform elementwise
#     Nmns = MAMresid.shape[2]
#     if Nmns > max_num_mns:
#         patch_ix = np.random.choice(Nmns, max_num_mns, replace=False)
#         MAMresid_downsampled = MAMresid[:, :, patch_ix]
#     else:
#         patch_ix = np.ones(Nmns).astype(bool)
#         MAMresid_downsampled = MAMresid
#     nullmncorrs = np.einsum('nmp,nl->lmp',
#                             MAMresid_downsampled.astype(np.float32),
#                             ycond_.astype(np.float32)) / n  # (Nsim, Nmodels, Npatches_ds)
#     nullmncorrs = np.arctanh(nullmncorrs)

#     # (5) null meta-analyzed correlation per (patch, simulate) = average over models
#     nullmncorrs_meta = nullmncorrs.mean(axis=1).T  # (Npatches_ds, Nsim)

#     # (6) per-patch std of the null meta-analyzed correlation across simulates
#     sds = nullmncorrs_meta.std(axis=1)  # (Npatches_ds,)
#     globalweights = 1 / sds

#     # (7) global statistic = inverse-variance-weighted sum of meta correlations
#     #     (over the same subsampled patches the null is defined on)
#     mncorrs_meta_downsampled = mncorrs_meta[patch_ix]  # (Npatches_ds,)
#     globalstat = ((mncorrs_meta_downsampled * globalweights)**2).sum()

#     # (8) null global statistics, one per simulate
#     nullglobalstats = ((nullmncorrs_meta * globalweights[:, None])**2).sum(axis=0)  # (Nsim,)

#     # (9) empirical global p-value
#     p = ((nullglobalstats >= globalstat).sum() + 1)/(len(nullglobalstats) + 1)
#     print(f'\033[32mP = {p}\033[0m')
#     if p <= 1/(Nnull + 1)+1e-10:
#         warnings.warn('global association p-value attained minimal possible value. '+\
#                 'Consider increasing Nnull')

#     # (10) per-patch FDRs. thresholds must be derived from the same (subsampled)
#     #      patches passed to empirical_fdrs, else the highest thresholds have zero
#     #      real detections (div by zero)
#     maxcorr = max(np.abs(mncorrs_meta_downsampled).max(), 0.001)
#     fdr_thresholds = np.arange(maxcorr/4, maxcorr, maxcorr/400)
#     fdr_vals = cna.tl._stats.empirical_fdrs(mncorrs_meta_downsampled, nullmncorrs_meta, fdr_thresholds)
#     fdrs = pd.DataFrame({
#         'threshold':fdr_thresholds,
#         'fdr':fdr_vals,
#         'num_detected': [(np.abs(mncorrs_meta)>t).sum() for t in fdr_thresholds]})

#     res = {'p':p, 'mncorrs':mncorrs_meta, 'fdrs':fdrs,
#             'globalstat':globalstat, 'nullglobalstats':nullglobalstats,
#             'weights':weights,
#             'nullmncorrs':nullmncorrs_meta,
#             }
#     return Namespace(**res)

def association(ds, y, sid_name, batches=None, covs=None, donorids=None, key_added='mncoef',
                return_full=False, ridges=None, cached_res=None,
                Nnull=10000, seed=0, make_umap=True,
                nsteps=None, show_progress=False, allow_low_sample_size=False,
                max_num_mns=200000, **kwargs):
    if seed is not None: np.random.seed(seed)

    # Check formats of inputs and figure out which samples have valid data
    batches, filter_samples = cna.tl._association.check_inputs(ds.select_model(0), y, sid_name, batches, covs, donorids, allow_low_sample_size)

    # Compute NAMs and filter to the appopriate samples and columns
    if cached_res is None:
        print('computing MAT') #TODO: rename MAM to MAT in code if we keep this nomenclature
        MAMs = []
        kepts = []
        for d in tqdm(ds.modelspecific_fingerprints(), total=ds.nmodels, ncols=100):
            MAM, kept, batches, covs, donorids, filter_samples = cna.tl._association.compute_nam_and_reindex(
                d, y, sid_name, batches, covs, donorids, filter_samples, nsteps, show_progress, **kwargs)
            MAMs.append(MAM)
            kepts.append(kept)
        kept = np.logical_and.reduce(kepts)
        
        for i in range(len(MAMs)):
            MAMs[i] = MAMs[i][ds.obs.index[kept]]

        # residualize NAMs
        MAMs_concat = pd.concat(MAMs, axis=1)
        MAMs_concat.columns = range(MAMs_concat.shape[1])
        res = cna.tl._nam._resid_nam(MAMs_concat,
                            covs[filter_samples] if covs is not None else covs,
                            batches[filter_samples] if batches is not None else batches,
                            npcs=1,
                            ridges=ridges,
                            show_progress=show_progress)
        MAMs_concat = res.namresid
    else:
        import copy
        res = copy.deepcopy(cached_res)
        MAMs_concat = res.namresid
        kept = res.kept

    print('performing association test')
    n_samples, n_total = MAMs_concat.shape
    Npatches = n_total // ds.nmodels
    MAMresid = MAMs_concat.values.reshape(n_samples, ds.nmodels, Npatches)
    res_, mncorrs, nullmncorrs, patch_ix = _association(
        MAMresid, res.M.values,
        y[filter_samples].values, batches[filter_samples].values,
        donorids[filter_samples].values if donorids is not None else None,
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
    
    # compute local FDRs
    def min_fdr_for_corr(ncorr):
        matching_fdrs = res.fdrs.loc[res.fdrs.threshold <= abs(ncorr)].fdr
        return matching_fdrs.min() if not matching_fdrs.empty else 1
    D.obs[f'{key_added}_fdr'] = D.obs[key_added].apply(min_fdr_for_corr)
    D.uns['vima_p'] = res.p
    D.uns['vima_pheno'] = y.name

    if return_full:
        return res, D
    else:
        return res.p, D