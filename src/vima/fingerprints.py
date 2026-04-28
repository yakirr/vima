import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import cna


class Fingerprints:
    def __init__(self, adata):
        self._adata = adata

    @classmethod
    def from_list(cls, ds):
        packed = ad.AnnData(obs=ds[0].obs.copy())
        packed.uns['n_fingerprints'] = len(ds)
        for i, d in enumerate(ds):
            packed.obsm[f'X_{i}'] = d.X
            packed.obsp[f'connectivities_{i}'] = d.obsp['connectivities']
            packed.obsp[f'distances_{i}'] = d.obsp['distances']
            packed.uns[f'neighbors_{i}'] = d.uns['neighbors']
        return cls(packed)

    def __len__(self):
        return self._adata.uns['n_fingerprints']

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, i):
        d = ad.AnnData(X=self._adata.obsm[f'X_{i}'], obs=self._adata.obs.copy())
        d.obsp['connectivities'] = self._adata.obsp[f'connectivities_{i}']
        d.obsp['distances'] = self._adata.obsp[f'distances_{i}']
        d.uns['neighbors'] = self._adata.uns[f'neighbors_{i}']
        return d

    def __repr__(self):
        n = len(self)
        npatches = len(self._adata)
        emb_dim = self._adata.obsm['X_0'].shape[1]
        return f'Fingerprints object with nmodels × npatches × latentdim = {n} × {npatches} × {emb_dim}.'

    @property
    def obs(self):
        return self._adata.obs

    def weighted_avg_graph(self, weights, kept, make_umap=True):
        M = kept.sum()
        obs = self[0].obs.iloc[kept].copy(deep=True)
        obs.index = obs.index.astype(str)
        D = ad.AnnData(X=np.random.randn(M, self[0].X.shape[1]), obs=obs)

        combined = sp.csr_matrix((M, M))
        combined_dist = sp.csr_matrix((M, M))
        for d, w in zip(self, weights):
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
            np.ones((len(self), len(self[0]))) / len(self),
            kept=np.ones(len(self[0]), dtype=bool),
            make_umap=make_umap,
        )

    def sample_pcs(self, sid_name='sid'):
        D = self.avg_graph(make_umap=False)
        NAM, _ = cna.tl.nam(D, sid_name)
        NAM -= NAM.mean(axis=0)
        NAM /= NAM.std(axis=0)
        U, _, _ = np.linalg.svd(NAM, full_matrices=False)
        return pd.DataFrame(U, index=NAM.index,
                            columns=[f'PC{i+1}' for i in range(U.shape[1])])

    def to_anndata(self):
        X = np.hstack([self._adata.obsm[f'X_{i}'] for i in range(len(self))])
        return ad.AnnData(X=X, obs=self._adata.obs.copy())

    def write_h5ad(self, path):
        self._adata.write_h5ad(path)

    @classmethod
    def read_h5ad(cls, path):
        return cls(ad.read_h5ad(path))
