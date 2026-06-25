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

def gaussian_perpatchll_test(obs, null, nnull=None, *, do_significance=True, ridge=0.0, chunk=2000):
    K, Nm, M = null.shape
    LOG2PI = np.log(2 * np.pi)
    eye = np.eye(Nm)
    obs = obs.astype(np.float64)
 
    LL_obs = np.empty(M)                                      # per-patch obs log-likelihood
    LL_null = np.empty((K, M)) if do_significance else None   # per-patch null log-likelihoods
 
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        nv = null[:, :, s:e].astype(np.float64)              # (K, Nm, b)
        mu = nv.mean(0)
        cen = nv - mu[None]
        Sc = np.einsum('kip,kjp->pij', cen, cen)             # full-K scatter (shared)
 
        # obs: scored against the first K-1 nulls -- drop the last null from the scatter
        w = cen[-1]
        Sc_obs = Sc - (K / (K - 1)) * np.einsum('ip,jp->pij', w, w)
        Sig_obs = Sc_obs / (K - 2)
        if ridge:
            Sig_obs = Sig_obs + ridge * eye
        Po = np.linalg.inv(Sig_obs)
        _, ldo = np.linalg.slogdet(Sig_obs)
        muo = (K * mu - nv[-1]) / (K - 1)
        do = obs[:, s:e] - muo
        maha_o = np.einsum('ip,pij,jp->p', do, Po, do)
        LL_obs[s:e] = -0.5 * (maha_o + Nm * LOG2PI + ldo)
 
        if do_significance:
            # nulls: leave-one-out via rank-1 downdate of the full-K covariance
            Sig = Sc / (K - 1)
            if ridge:
                Sig = Sig + ridge * eye
            P = np.linalg.inv(Sig)
            _, ld = np.linalg.slogdet(Sig)
            t = np.einsum('kip,pij->kjp', cen, P)
            q = np.einsum('kjp,kjp->kp', t, cen)             # (K, b)
            cg = np.minimum(K * q / (K - 1) ** 2, 1 - 1e-9)
            maha_loo = (K / (K - 1)) ** 2 * (K - 2) * (q / (K - 1)) / (1 - cg)
            ld_loo = ld[None] + Nm * np.log((K - 1) / (K - 2)) + np.log(np.maximum(1 - cg, 1e-300))
            LL_null[:, s:e] = -0.5 * (maha_loo + Nm * LOG2PI + ld_loo)
 
    if not do_significance:
        return LL_obs
 
    LL_obs_total = LL_obs.sum()
    LL_null_total = LL_null.sum(1)                            # (K,)
    global_p = (1 + np.sum(LL_null_total <= LL_obs_total)) / (1 + K)   # lower tail = signal
    return float(global_p), LL_obs, LL_null

def _stats(R, Y):
    # R: (n, Nmodels, b)   Y: (n, K)  ->  (K, Nmodels, b) of mean_n(Y * R)
    return np.tensordot(Y, R, axes=([0], [0])) / R.shape[0]
 
def screen_all_patches(MAMresid, ycond, ycond_null, patch_chunk=10000):
    """PASS 1: per-patch observed log-likelihoods for ALL patches (do_significance=False).
       ycond_null is (n, Nnull_screen), e.g. Nnull_screen = 500-1000."""
    n, Nm, Npatches = MAMresid.shape
    corrs = np.empty((Nm, Npatches))
    LL_obs = np.empty(Npatches)
    for s in tqdm(range(0, Npatches, patch_chunk), desc='computing LLs for all patches'):
        e = min(s + patch_chunk, Npatches)
        R = np.asarray(MAMresid[:, :, s:e], dtype=np.float64)          # (n, Nm, b)
        obs_c  = (ycond[:, None, None] * R).mean(0)                    # (Nm, b)
        null_c = _stats(R, ycond_null)                                # (Nnull_screen, Nm, b)
        LL_obs[s:e] = gaussian_perpatchll_test(obs_c, null_c, do_significance=False)
        corrs[:,s:e] = obs_c
        del R, null_c
    return LL_obs, corrs
 
 
def significance_on_subset(MAMresid, ycond, ycond_null_big, n_patches=5000,
                           rng=None, sub_chunk=1000):
    """PASS 2: random patch subset + a large null set (ycond_null_big is (n, Nnull_big),
       e.g. Nnull_big = 10000). Patches are processed in sub-chunks so the big
       (Nnull_big, Nm, sub_chunk) correlation tensor is bounded; the per-null totals
       are accumulated to form the global p-value."""
    n, Nm, Npatches = MAMresid.shape
    Kbig = ycond_null_big.shape[1]
    rng = np.random.default_rng() if rng is None else rng
    if n_patches > Npatches:
        print(f'WARNING: cannot subsample to {n_patches} patches because there are only {Npatches} patches.')
        sel = np.arange(Npatches)
    else:
        sel = np.sort(rng.choice(Npatches, size=n_patches, replace=False))
 
    LL_obs = np.empty(n_patches)
    LL_null = np.empty((Kbig, n_patches))
    for a in tqdm(range(0, n_patches, sub_chunk), desc='calibrating LL statistics'):
        b = min(a + sub_chunk, n_patches)
        R = np.asarray(MAMresid[:, :, sel[a:b]], dtype=np.float64)     # (n, Nm, sb)
        obs_c  = (ycond[:, None, None] * R).mean(0)                    # (Nm, sb)
        null_c = _stats(R, ycond_null_big)                            # (Nnull_big, Nm, sb)
        _, llo, lln = gaussian_perpatchll_test(obs_c, null_c, do_significance=True)
        LL_obs[a:b] = llo
        LL_null[:, a:b] = lln
        del R, null_c
 
    obs_total = LL_obs.sum()
    null_total = LL_null.sum(1)                                        # (Nnull_big,)
    global_p = (1 + np.sum(null_total <= obs_total)) / (1 + Kbig)      # lower tail = signal
    return sel, float(global_p), LL_obs, LL_null
 
 
def empirical_fdr(LL_obs, LL_null, thresholds):
    """Lower-tail detection (LL <= threshold). LL_null is (Nnull, n_patches_subset),
       a random subset that estimates the genome-wide null-LL distribution.
       FDR(t) = (expected null fraction below t) * Npatches_obs / (#obs below t)."""
    thr = np.asarray(thresholds, float)
    null_frac = (LL_null[:, :, None] <= thr[None, None, :]).mean(axis=(0, 1))   # (T,)
    R = (LL_obs[:, None] <= thr[None, :]).sum(0)                                # (T,)
    fdr = np.where(R > 0, null_frac * LL_obs.size / np.maximum(R, 1), 0.0)
    return np.minimum.accumulate(fdr[::-1])[::-1]                               # monotone
 

def _association(MAMresid, M, y, batches, donorids, rng, Nnullscreen=500, Nnullfull=10000,
                 max_num_mns=5_000, show_progress=False):
    # prep data
    y = (y - y.mean())/y.std()
    n = len(y)
    ycond = M.dot(y)
    ycond /= ycond.std(axis=0)

    # make null phenotypes
    if donorids is not None:
        yscreen = cna.tl._stats.grouplevel_permutation(donorids, y, Nnullscreen)
        ybig = cna.tl._stats.grouplevel_permutation(donorids, y, Nnullfull)
    else:
        yscreen = cna.tl._stats.conditional_permutation(batches, y, Nnullscreen)
        ybig = cna.tl._stats.conditional_permutation(batches, y, Nnullfull)
    ycond_screen = M.dot(yscreen); ycond_screen /= ycond_screen.std(axis=0)
    ycond_big = M.dot(ybig); ycond_big /= ycond_big.std(axis=0)

    # MAMresid: (n, Nmodels, Npatches)
    # get microniche coefficients and weights
    LL_all, mncorrs = screen_all_patches(MAMresid, ycond, ycond_screen, patch_chunk=10000)
    sel, p, LLo, LLn = significance_on_subset(MAMresid, ycond, ycond_big, n_patches=max_num_mns, rng=rng)
    print(f'\033[32mP = {p}\033[0m')
    if p <= 1/(Nnullfull + 1)+1e-10:
        warnings.warn('global association p-value attained minimal possible value. '+\
                'Consider increasing Nnull')
        
    thr = np.quantile(LLo, np.arange(0.01, 1, 0.01))
    fdrs = empirical_fdr(LLo, LLn, thr)

    weights = (mncorrs**2) / (mncorrs**2).sum(axis=0)
    mncorrs_meta = mncorrs.mean(axis=0)

    fdrs = pd.DataFrame({
        'll_threshold':thr,
        'fdr':fdrs})

    res = {'p':p, 'mncorrs':mncorrs_meta, 'LLs':LL_all, 'fdrs':fdrs,
            'globalstat':LLo.sum(), 'nullglobalstats':LLn.sum(axis=1),
            'nullpermnLLs':LLn,
            'weights':weights,
            'permodel_mncorrs':mncorrs,
            'MAMres':MAMresid,
            'ycond':ycond
            }
    return Namespace(**res), mncorrs, None, None


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
                Nnull=10000, seed=0, make_umap=True,
                nsteps=None, show_progress=False, allow_low_sample_size=False,
                max_num_mns=5000, **kwargs):
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
    res_, mncorrs, nullmncorrs, patch_ix = _association(
        MAMresid, res.M.values,
        y[filter_samples].values, batches[filter_samples].values,
        donorids[filter_samples].values if donorids is not None else None,
        rng,
        max_num_mns=max_num_mns,
        show_progress=show_progress, Nnullfull=Nnull,
        **kwargs)
    res.__dict__.update(vars(res_)) # add info from from res_ to res
    res.kept = kept
    
    # make anndata with results
    D = ds.weighted_avg_graph(res.weights, kept, make_umap=make_umap)
    if key_added in D.obs:
        warnings.warn(f"Key '{key_added}' already exists in d.obs. Overwriting.")
    D.obs['LL'] = res.LLs
    D.obs[key_added] = res.mncorrs
    D.obsm['permodel_mncorrs'] = pd.DataFrame(res.permodel_mncorrs.T,
                                              columns=[f'model{i}' for i in range(1, ds.nmodels+1)],
                                              index=D.obs.index)
    
    # compute local FDRs
    def min_fdr_for_corr(ll):
        matching_fdrs = res.fdrs.loc[res.fdrs.ll_threshold >= ll].fdr
        return matching_fdrs.min() if not matching_fdrs.empty else 1
    D.obs[f'{key_added}_fdr'] = D.obs.LL.apply(min_fdr_for_corr)
    D.uns['vima_p'] = res.p
    D.uns['vima_pheno'] = y.name

    if return_full:
        return res, D
    else:
        return res.p, D