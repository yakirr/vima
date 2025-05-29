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
pb = lambda x: tqdm(x, ncols=100)

def anndata(patchmeta, Z, var_names=None, use_rep='X', n_comps=10, **kwargs):
    d = ad.AnnData(Z)
    if var_names is not None:
        d.var_names = var_names
    d.obs = patchmeta

    if use_rep == 'X_pca':
        sc.tl.pca(d, n_comps=min(n_comps, Z.shape[1]-1))

    sc.pp.neighbors(d, use_rep=use_rep, **kwargs)

    return d

def apply(models, P, batch_size=1000):
    P.pytorch_mode()
    P.augmentation_off()
    for model in models:
        model.eval()
    
    eval_loader = DataLoader(
        dataset=P,
        batch_size=batch_size,
        shuffle=False)

    Zs = {modelid: [] for modelid in range(len(models))}
    with torch.no_grad():
        for batch in pb(eval_loader):
            for modelid, model in enumerate(models):
                Zs[modelid].append(model.embedding(batch).detach().cpu().numpy())

    return np.array([
        np.concatenate(Z) for Z in Zs.values()])

def latentreps(models, P, use_rep='X', n_comps=100, **kwargs):
    print('applying models')
    Zs = apply(models, P)
    print('computing nearest-neighbor graphs')
    return [
        anndata(P.meta, Z, use_rep=use_rep, n_comps=n_comps, **kwargs)
        for Z in pb(Zs)]

def _association(MAMresid, M, Nmodels, y, batches, donorids, ks=None, Nnull=1000, show_progress=False):
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

    # get microniche coefficients and weights
    mncorrs = (ycond[:,None]*MAMresid).mean(axis=0)
    mncorrs = mncorrs.reshape((Nmodels, -1))
    weights = (mncorrs**2) / (mncorrs**2).sum(axis=0)
    nullmncorrs = MAMresid.astype(np.float32).T.dot(ycond_.astype(np.float32)) / n
    nullmncorrs = nullmncorrs.reshape(Nmodels, -1, Nnull).transpose(2, 0, 1) # simulates X models x patches
    
    # computed meta-analyzed mn coefficients and global test statistics
    globalstat = ((mncorrs**4).sum(axis=0)/(mncorrs**2).sum(axis=0)).mean()
    mncorrs_meta = (mncorrs**3).sum(axis=0)/(mncorrs**2).sum(axis=0)
    nullglobalstats = ((nullmncorrs**4).sum(axis=1)/(nullmncorrs**2).sum(axis=1)).mean(axis=1)
    nullmncorrs_meta = ((nullmncorrs**3).sum(axis=1)/(nullmncorrs**2).sum(axis=1)).T

    # compute global p-vaule
    p = ((nullglobalstats >= globalstat).sum() + 1)/(len(nullglobalstats) + 1)
    print(f'P = {p}')
    if p <= 1/(Nnull + 1)+1e-10:
        warnings.warn('global association p-value attained minimal possible value. '+\
                'Consider increasing Nnull')
        
    shape, loc, scale = st.gamma.fit(nullglobalstats)
    gamma_p = 1 - st.gamma.cdf(globalstat, shape, loc=loc, scale=scale)
    print(f'Gamma p-value = {gamma_p}')
    
    maxcorr = max(np.abs(mncorrs_meta).max(), 0.001)
    fdr_thresholds = np.arange(maxcorr/4, maxcorr, maxcorr/400)
    fdr_vals = cna.tl._stats.empirical_fdrs(mncorrs_meta, nullmncorrs_meta, fdr_thresholds)
    fdrs = pd.DataFrame({
        'threshold':fdr_thresholds,
        'fdr':fdr_vals,
        'num_detected': [(np.abs(mncorrs)>t).sum() for t in fdr_thresholds]})

    res = {'p':p, 'gamma_p':gamma_p, 'mncorrs':mncorrs_meta, 'fdrs':fdrs,
            'globalstat':globalstat, 'nullglobalstats':nullglobalstats,
            'weights':weights,
            'nullmncorrs':nullmncorrs_meta,
            }
    return Namespace(**res)

def avg_graph(ds, weights, kept, make_umap=True):
    M = kept.sum()
    D = sc.AnnData(X=np.random.randn(M, ds[0].X.shape[1]),
                   obs=ds[0].obs.iloc[kept].copy(deep=True))

    combined = sp.csr_matrix((M, M))
    combined_dist = sp.csr_matrix((M, M))
    for d, w in zip(ds, weights):
        row_scaling = sp.diags(w)
        combined += row_scaling @ d.obsp['connectivities'][kept, :][:, kept]
        combined_dist += row_scaling @ d.obsp['distances'][kept, :][:, kept]
    D.obsp['connectivities'] = combined
    D.obsp['distances'] = combined_dist    
    
    D.uns['neighbors'] = {
        'connectivities_key': 'connectivities',
        'distances_key': 'distances',
        'params': {
            'method': 'custom',
            'metric': 'euclidean'  # or whatever is appropriate
        }
    }
    if make_umap:
        sc.tl.umap(D, neighbors_key='neighbors')

    return D

def association(ds, y, sid_name, batches=None, covs=None, donorids=None, key_added='mncoef',
                return_full=False, ridges=None, Nnull=10000, seed=0, make_umap=True,
                nsteps=None, show_progress=False, allow_low_sample_size=False, **kwargs):
    if seed is not None: np.random.seed(seed)
    
    # Check that all ds have identical metadata
    if not all(ds[0].obs.equals(d.obs) for d in ds):
        raise ValueError("All datasets must have identical metadata (obs).")

    # Check formats of inputs and figure out which samples have valid data
    batches, filter_samples = cna.tl._association.check_inputs(ds[0], y, sid_name, batches, covs, donorids, allow_low_sample_size)

    # Compute NAMs and filter to the appopriate samples and columns
    print('computing MAMs')
    MAMs = []
    kepts = []
    for d in ds:
        MAM, kept, batches, covs, donorids, filter_samples = cna.tl._association.compute_nam_and_reindex(
            d, y, sid_name, batches, covs, donorids, filter_samples, nsteps, show_progress, **kwargs)
        MAMs.append(MAM)
        kepts.append(kept)
    kept = np.logical_and.reduce(kepts)
    for i in range(len(MAMs)):
        MAMs[i] = MAMs[i][ds[0].obs.index[kept]]

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

    print('performing association test')
    res_ = _association(
        MAMs_concat.values, res.M.values, len(ds),
        y[filter_samples].values, batches[filter_samples].values,
        donorids[filter_samples].values if donorids is not None else None,
        show_progress=show_progress, Nnull=Nnull,
        **kwargs)
    res.__dict__.update(vars(res_)) # add info from from res_ to res
    res.kept = kept
    
    # make anndata with results
    D = avg_graph(ds, res.weights, kept, make_umap=make_umap)
    if key_added in D.obs:
        warnings.warn(f"Key '{key_added}' already exists in d.obs. Overwriting.")
    D.obs[key_added] = res.mncorrs
    
    # compute local FDRs
    def min_fdr_for_corr(ncorr):
        matching_fdrs = res.fdrs.loc[res.fdrs.threshold <= abs(ncorr)].fdr
        return matching_fdrs.min() if not matching_fdrs.empty else 1
    D.obs[f'{key_added}_fdr'] = D.obs[key_added].apply(min_fdr_for_corr)

    if return_full:
        return res, D
    else:
        return res.p, D