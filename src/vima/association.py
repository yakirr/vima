import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import torch
from torch.utils.data import DataLoader
import cna
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

def anndata(patchmeta, Z, var_names=None, use_rep='X', n_comps=10):
    d = ad.AnnData(Z)
    if var_names is not None:
        d.var_names = var_names
    d.obs = patchmeta

    if use_rep == 'X_pca':
        sc.tl.pca(d, n_comps=min(n_comps, Z.shape[1]-1))

    print('running UMAP')
    sc.pp.neighbors(d, use_rep=use_rep)
    sc.tl.umap(d)
    print(f'done')

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

def latentrep(model, P):
    return anndata(P.meta,
                apply(model, P))

def association(d, y, sid_name, batches=None, covs=None, donorids=None, key_added='mncoef',
                return_full=False, Nnull=10000, fdr=0.1, seed=0, **kwargs):
    if seed is not None: np.random.seed(seed)
    
    res = cna.tl.association(d, y, sid_name, donorids=donorids, covs=covs, batches=batches,
                             key_added=key_added, Nnull=Nnull, return_full=True, **kwargs)
    
    print(f'P = {res.p}, used {res.k} MAM-PCs')
    
    mncoef = d.obs[key_added]
    fdrs = d.obs[f'{key_added}_fdr']
    nsig = (fdrs <= fdr).sum()
    print(f'Found {nsig} microniches at FDR {int(fdr*100)}%')
    d.obs['sig_mncoef'] = mncoef * (fdrs <= fdr)

    if return_full:
        return res
    else:
        return res.p