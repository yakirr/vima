import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import multianndata as md
import torch
from torch.utils.data import DataLoader
import cna
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

def anndata(patchmeta, Z, samplemeta, var_names=None, use_rep='X', n_comps=10, sampleid='sid'):
    d = ad.AnnData(Z)
    if var_names is not None:
        d.var_names = var_names
    d.obs = patchmeta

    if use_rep == 'X_pca':
        sc.tl.pca(d, n_comps=min(n_comps, Z.shape[1]-1))

    print('running UMAP')
    sc.pp.neighbors(d, use_rep=use_rep)
    sc.tl.umap(d)

    samplemeta.index = samplemeta.index.astype(str)
    d.obs.sid = d.obs.sid.astype(str)
    d = md.MultiAnnData(d)
    d.samplem = samplemeta
    d.sampleid = sampleid
    print(f'built MultiAnnData object with {sampleid} as the unit of analysis')

    return d

def apply(model, P, embedding=None, batch_size=1000):
    if embedding is None:
        embedding = model.embedding

    P.pytorch_mode()
    P.augmentation_off()
    model.eval()
    eval_loader = DataLoader(
        dataset=P,
        batch_size=batch_size,
        shuffle=False)

    Z = []
    with torch.no_grad():
        for batch in pb(eval_loader):
            Z.append(embedding(batch).detach().cpu().numpy())

    return np.concatenate(Z)

def latentrep(model, P, samplemeta):
    return anndata(P.meta,
                apply(model, P),
                samplemeta[samplemeta.index.isin(P.meta.sid.unique())],
                sampleid='sid')

def association(d, pheno, fdr=0.1, force_recompute=True, covs=None, Nnull=10000, seed=0, **kwargs):
    if seed is not None: np.random.seed(seed)
    cna.tl.nam(d, force_recompute=force_recompute)
    d.samplem['case'] = pheno
    if covs is not None:
        d.samplem[covs.columns] = covs
        
    res = cna.tl.association(d, d.samplem.case, donorids=d.samplem.donor.values, covs=covs, Nnull=Nnull, **kwargs)
    print(f'P = {res.p}, used {res.k} MAM-PCs')
    d.obs['mncoeff'] = res.ncorrs
    if res.fdrs.fdr.min() <= fdr:
        print(f'Found {res.fdrs[res.fdrs.fdr < 0.1].iloc[0].num_detected} microniches at FDR {int(fdr*100)}%')
        d.obs['sig_mncoeff'] = res.ncorrs * (np.abs(res.ncorrs) > res.fdrs[res.fdrs.fdr < fdr].iloc[0].threshold)
    else:
        print(f'No microniches found at FDR {int(fdr*100)}%')
        d.obs['sig_mncoeff'] = 0
    d.samplem.loc[~np.isnan(d.samplem.case), 'yhat'] = res.yresid_hat

    return res