import matplotlib.pyplot as plt
import scanpy as sc


def plot_association(D, key='mncoef', fdr_thresh=0.1, ax=None, show=True, **kwargs):
    """
    Plot the microniche UMAP, highlighting significant microniches.

    Microniches passing `fdr_thresh` are colored by their signed association
    coefficient (red positive, blue negative); the rest are shown in gray.

    Parameters
    ----------
    D
        Association result AnnData from `association`.
    key
        Coefficient column in ``D.obs`` (with matching ``{key}_fdr``).
    fdr_thresh
        FDR cutoff for calling a microniche significant.
    """
    if ax is None:
        ax = plt.gca()
    sig = D.obs[key].where(D.obs[f'{key}_fdr'] <= fdr_thresh, 0)
    D = D.copy()
    D.obs['_sig'] = sig
    sc.pl.umap(D, ax=ax, show=False, **kwargs)

    if (D.obs._sig != 0).sum() > 0:
        sc.pl.umap(D[D.obs._sig != 0], color='_sig', cmap='seismic', vmin=-1, vmax=1,
               ax=ax, title=f'{(D.obs._sig != 0).sum()} microniches at FDR {fdr_thresh*100:.0f}%',
               frameon=False,
               show=False, **kwargs)
    else:
        plt.title(f'No significant microniches at FDR {fdr_thresh*100:.0f}%')
        
    if show:
        plt.show()
