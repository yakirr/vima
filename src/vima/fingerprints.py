import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import cna
import copy
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)


class _ObsmView:
    """Filtered write-through proxy over an AnnData obsm mapping."""
    def __init__(self, obsm, exclude):
        self._obsm = obsm
        self._exclude = exclude

    def _visible(self, key):
        return key not in self._exclude

    def __setitem__(self, key, value):   self._obsm[key] = value
    def __delitem__(self, key):          del self._obsm[key]
    def __getitem__(self, key):
        if not self._visible(key): raise KeyError(key)
        return self._obsm[key]
    def __contains__(self, key):  return self._visible(key) and key in self._obsm
    def __iter__(self):           return (k for k in self._obsm if self._visible(k))
    def __repr__(self):           return f"_ObsmView with keys: {', '.join(self.keys())}"
    def keys(self):               return [k for k in self._obsm if self._visible(k)]
    def values(self):             return [self._obsm[k] for k in self._obsm if self._visible(k)]
    def items(self):              return [(k, self._obsm[k]) for k in self._obsm if self._visible(k)]


class Fingerprints:
    """
    Ensemble of per-model patch embeddings and neighbor graphs for multiple
    models.

    Packs all informationinto a single AnnData that contains each model's
    embedding (``X_i``), neighbor graph, and shared patch metadata. Produced
    by `latentreps` and consumed by `association`. Patch metadata is exposed
    via ``.obs``; auxiliary per-patch matrices (e.g. cell type abundances)
    via ``.obsm``; subsetting and ``read_h5ad``/``write_h5ad`` behave like
    AnnData.
    """

    def __init__(self, adata):
        self._adata = adata

    @classmethod
    def from_list(cls, ds):
        """
        Pack a list of per-model AnnData embeddings and graphs into a Fingerprints.

        Parameters
        ----------
        ds
            One AnnData per model, sharing patch metadata, each carrying an
            embedding and its neighbor graph.
        """
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
        """Number of models in the ensemble."""
        return self._adata.uns['n_models']

    def __getitem__(self, key):
        return Fingerprints(self._adata[key].copy())

    def select_model(self, i):
        """Return model `i`'s embedding and neighbor graph as an AnnData."""
        d = ad.AnnData(X=self._adata.obsm[f'X_{i}'], obs=self._adata.obs.copy())
        d.obsp['connectivities'] = self._adata.obsp[f'connectivities_{i}']
        d.obsp['distances'] = self._adata.obsp[f'distances_{i}']
        d.obsm = copy.deepcopy(self._adata.obsm)
        d.uns['neighbors'] = self._adata.uns[f'neighbors_{i}']
        return d

    def modelspecific_fingerprints(self):
        """Iterate over each model's embedding and graph as an AnnData."""
        return (self.select_model(i) for i in range(self.nmodels))

    def __repr__(self):
        n = self.nmodels
        npatches = len(self._adata)
        emb_dim = self._adata.obsm['X_0'].shape[1]
        return f'Fingerprints object with nmodels × npatches × latentdim = {n} × {npatches} × {emb_dim}.'

    @property
    def obs(self):
        """Per-patch metadata."""
        return self._adata.obs

    @property
    def obsm(self):
        """Auxiliary per-patch matrices (excluding the per-model embeddings)."""
        exclude = {f'X_{i}' for i in range(self.nmodels)}
        return _ObsmView(self._adata.obsm, exclude)

    def weighted_avg_graph(self, weights, kept, make_umap=True):
        """
        Combine the per-model microniche graphs into one weighted-average graph.

        Averages the per-model connectivity and distance graphs over the retained
        patches, weighting each model's contribution per microniche, and returns
        the result as a single AnnData suitable for downstream visualization.

        Parameters
        ----------
        weights
            Per-model, per-microniche mixing weights.
        kept
            Boolean mask selecting which patches to include.

        Returns
        -------
        AnnData
            Microniches with the combined neighbor graph (and a UMAP if
            `make_umap`).
        """
        M = kept.sum()
        obs = self.obs.iloc[kept].copy(deep=True)
        obs.index = obs.index.astype(str)
        D = ad.AnnData(X=np.random.randn(M, self.select_model(0).X.shape[1]),
                       obs=obs,
                       obsm={k:v[kept] for k, v in self.obsm.items()})

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
            print('Computing UMAP...')
            sc.tl.umap(D, neighbors_key='neighbors')
        return D

    def avg_graph(self, make_umap=True):
        """Combine the per-model graphs with equal weights over all patches."""
        return self.weighted_avg_graph(
            np.ones((self.nmodels, len(self._adata))) / self.nmodels,
            kept=np.ones(len(self._adata), dtype=bool),
            make_umap=make_umap,
        )

    def compute_nngs(self, **kwargs):
        """Recompute and store each model's nearest-neighbor graph."""
        for i in pb(range(self.nmodels)):
            d = self.select_model(i)
            sc.pp.neighbors(d, **kwargs)
            self._adata.obsp[f'connectivities_{i}'] = d.obsp['connectivities']
            self._adata.obsp[f'distances_{i}'] = d.obsp['distances']
            self._adata.uns[f'neighbors_{i}'] = d.uns['neighbors']
        
    def sample_pcs(self, sid_name='sid'):
        """
        Compute principal components of the sample-by-microniche abundance matrix.

        Standardizes each microniche's abundance across samples, then returns the
        left singular vectors as per-sample PC scores.

        Parameters
        ----------
        sid_name
            Column in ``.obs`` giving each patch's sample ID.

        Returns
        -------
        DataFrame
            Samples by PC.
        """
        D = self.avg_graph(make_umap=False)
        NAM, _ = cna.tl.nam(D, sid_name)
        NAM -= NAM.mean(axis=0)
        NAM /= NAM.std(axis=0)
        U, _, _ = np.linalg.svd(NAM, full_matrices=False)
        return pd.DataFrame(U, index=NAM.index,
                            columns=[f'PC{i+1}' for i in range(U.shape[1])])
    
    def mn_pcs(self, sid_name='sid'):
        """
        Compute principal components of the sample-by-microniche abundance matrix,
        as per-microniche loadings.

        Standardizes each microniche's abundance across samples, then returns the
        right singular vectors as per-microniche PC scores.

        Parameters
        ----------
        sid_name
            Column in ``.obs`` giving each patch's sample ID.

        Returns
        -------
        DataFrame
            Microniches by PC.
        """
        D = self.avg_graph(make_umap=False)
        NAM, _ = cna.tl.nam(D, sid_name)
        NAM -= NAM.mean(axis=0)
        NAM /= NAM.std(axis=0)
        _, _, VT = np.linalg.svd(NAM, full_matrices=False)
        print(VT.shape)
        return pd.DataFrame(VT.T, index=NAM.columns,
                            columns=[f'PC{i+1}' for i in range(VT.shape[0])])

    def to_anndata(self):
        """Concatenate all per-model embeddings into a single AnnData."""
        X = np.hstack([self._adata.obsm[f'X_{i}'] for i in range(self.nmodels)])
        return ad.AnnData(X=X, obs=self._adata.obs.copy())

    def write_h5ad(self, path):
        """Write the fingerprints to an ``.h5ad`` file."""
        self._adata.write_h5ad(path)

    @classmethod
    def read_h5ad(cls, path):
        """Load fingerprints previously saved with `write_h5ad`."""
        return cls(ad.read_h5ad(path))
