import matplotlib.pyplot as plt
import scanpy as sc


def plot_association(D, key='mncoef', fdr_thresh=0.1, ax=None, show=True, **kwargs):
    if ax is None:
        ax = plt.gca()
    sig = D.obs[key].where(D.obs[f'{key}_fdr'] <= fdr_thresh, 0)
    D = D.copy()
    D.obs['_sig'] = sig
    sc.pl.umap(D, ax=ax, show=False, **kwargs)
    sc.pl.umap(D[D.obs._sig != 0], color='_sig', cmap='seismic', vmin=-1, vmax=1,
               ax=ax, title=f'{(D.obs._sig != 0).sum()} microniches at FDR {fdr_thresh*100:.0f}%',
               show=False, **kwargs)
    if show:
        plt.show()
