import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import cna
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)


class Fingerprints:
    def __init__(self, adata):
        self._adata = adata

    @classmethod
    def from_list(cls, ds):
        packed = ad.AnnData(obs=ds[0].obs.copy())
        packed.uns['n_models'] = len(ds)
        for i, d in enumerate(ds):
            packed.obsm[f'X_{i}'] = d.X
            packed.obsp[f'connectivities_{i}'] = d.obsp['connectivities']
            packed.obsp[f'distances_{i}'] = d.obsp['distances']
            packed.uns[f'neighbors_{i}'] = d.uns['neighbors']
        return cls(packed)

    def __len__(self):
        return len(self._adata)

    @property
    def nmodels(self):
        return self._adata.uns['n_models']

    def __getitem__(self, key):
        return Fingerprints(self._adata[key].copy())

    def select_model(self, i):
        d = ad.AnnData(X=self._adata.obsm[f'X_{i}'], obs=self._adata.obs.copy())
        d.obsp['connectivities'] = self._adata.obsp[f'connectivities_{i}']
        d.obsp['distances'] = self._adata.obsp[f'distances_{i}']
        d.uns['neighbors'] = self._adata.uns[f'neighbors_{i}']
        return d

    def modelspecific_fingerprints(self):
        return (self.select_model(i) for i in range(self.nmodels))

    def __repr__(self):
        n = self.nmodels
        npatches = len(self._adata)
        emb_dim = self._adata.obsm['X_0'].shape[1]
        return f'Fingerprints object with nmodels × npatches × latentdim = {n} × {npatches} × {emb_dim}.'

    @property
    def obs(self):
        return self._adata.obs

    def weighted_avg_graph(self, weights, kept, make_umap=True):
        M = kept.sum()
        obs = self.select_model(0).obs.iloc[kept].copy(deep=True)
        obs.index = obs.index.astype(str)
        D = ad.AnnData(X=np.random.randn(M, self.select_model(0).X.shape[1]), obs=obs)

        combined = sp.csr_matrix((M, M))
        combined_dist = sp.csr_matrix((M, M))
        for d, w in zip(self.modelspecific_fingerprints(), weights):
            row_scaling = sp.diags(w)
            combined += row_scaling @ d.obsp['connectivities'][kept, :][:, kept]
            combined_dist += row_scaling @ d.obsp['distances'][kept, :][:, kept]
        D.obsp['connectivities'] = combined
        D.obsp['distances'] = combined_dist
        D.uns['neighbors'] = {
            'connectivities_key': 'connectivities',
            'distances_key': 'distances',
            'params': {'method': 'umap', 'metric': 'euclidean',
                       'n_neighbors': 15, 'use_rep': 'X', 'n_pcs': None},
        }
        if make_umap:
            sc.tl.umap(D, neighbors_key='neighbors')
        return D

    def avg_graph(self, make_umap=True):
        return self.weighted_avg_graph(
            np.ones((self.nmodels, len(self._adata))) / self.nmodels,
            kept=np.ones(len(self._adata), dtype=bool),
            make_umap=make_umap,
        )

    def compute_nngs(self, **kwargs):
        for i in pb(range(self.nmodels)):
            d = self.select_model(i)
            sc.pp.neighbors(d, **kwargs)
            self._adata.obsp[f'connectivities_{i}'] = d.obsp['connectivities']
            self._adata.obsp[f'distances_{i}'] = d.obsp['distances']
            self._adata.uns[f'neighbors_{i}'] = d.uns['neighbors']

    def sample_pcs(self, sid_name='sid'):
        D = self.avg_graph(make_umap=False)
        NAM, _ = cna.tl.nam(D, sid_name)
        NAM -= NAM.mean(axis=0)
        NAM /= NAM.std(axis=0)
        U, _, _ = np.linalg.svd(NAM, full_matrices=False)
        return pd.DataFrame(U, index=NAM.index,
                            columns=[f'PC{i+1}' for i in range(U.shape[1])])

    def to_anndata(self):
        X = np.hstack([self._adata.obsm[f'X_{i}'] for i in range(self.nmodels)])
        return ad.AnnData(X=X, obs=self._adata.obs.copy())

    def write_h5ad(self, path):
        self._adata.write_h5ad(path)

    @classmethod
    def read_h5ad(cls, path):
        return cls(ad.read_h5ad(path))
