import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _select_features(features, group_a, group_b, n_top, n_bottom, markers):
    if markers is not None:
        features = features[markers]
    group_a = np.asarray(group_a, dtype=bool)
    group_b = ~group_a if group_b is None else np.asarray(group_b, dtype=bool)
    diffs = (features[group_a].median() - features[group_b].median()).sort_values(ascending=False)
    toplot = list(diffs.index[:n_top])
    if n_bottom > 0:
        toplot = toplot + list(diffs.index[-n_bottom:])
    return features, group_a, group_b, toplot


def plot_features(
    features,
    group_a,
    group_b=None,
    n_top=5,
    n_bottom=5,
    markers=None,
    labels=None,
    kind='violin',
    ax=None,
    show=True,
    **kwargs,
):
    """
    Plot patch-level feature distributions for two groups side by side.

    Parameters
    ----------
    features
        Patch-by-feature values (e.g. cell type abundances or expression).
    group_a
        Boolean array over patches selecting the first group.
    group_b
        Boolean array selecting the second group; defaults to ``~group_a``.
    n_top, n_bottom
        Number of features most enriched in each group to show; ignored when
        `markers` is given.
    markers
        Explicit features to plot, overriding `n_top`/`n_bottom`.
    labels
        ``[label_a, label_b]`` for the legend.
    kind
        'violin' (default), 'box', or 'swarm'.
    """
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = ['a', 'b']

    features, group_a, group_b, toplot = _select_features(
        features, group_a, group_b, n_top, n_bottom, markers)

    mask = group_a | group_b
    df = features.loc[mask, toplot].copy()
    df['status'] = np.where(group_a[mask], labels[0], labels[1])
    df = df.melt(id_vars='status', value_vars=toplot, var_name='marker', value_name='value')

    plot_fn = {'violin': sns.violinplot, 'box': sns.boxplot, 'swarm': sns.swarmplot}.get(kind)
    if plot_fn is None:
        raise ValueError(f'kind must be "violin" or "box"; got {kind!r}')
    plot_kwargs = {'split': True, 'density_norm': 'count', 'inner': 'quart'} if kind == 'violin' else {}
    plot_kwargs.update(kwargs)
    plot_fn(data=df, x='marker', y='value', hue='status', order=toplot, ax=ax, **plot_kwargs)
    if show:
        plt.show()


def plot_features_by_sample(
    features,
    group_a,
    group_b=None,
    *,
    sample_key,
    n_top=10,
    n_bottom=0,
    markers=None,
    labels=None,
    ax=None,
    show=True,
    connect=True,
    **kwargs,
):
    """
    Swarmplot of sample-averaged feature values for two groups.

    Each dot is one sample's mean feature value across its patches in that
    group; samples with no patches in a group are omitted from it. This shows
    per-sample effects that patch-level plots can obscure.

    Parameters
    ----------
    features
        Patch-by-feature values.
    group_a
        Boolean array over patches selecting the first group.
    group_b
        Boolean array selecting the second group; defaults to ``~group_a``.
    sample_key
        Series over patches mapping each to its sample ID.
    n_top, n_bottom
        Number of features most enriched in each group to show; ignored when
        `markers` is given.
    markers
        Explicit features to plot, overriding `n_top`/`n_bottom`.
    labels
        ``[label_a, label_b]`` for the legend.
    connect
        Draw lines linking each sample's two group means.
    """
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = ['a', 'b']

    features, group_a, group_b, toplot = _select_features(
        features, group_a, group_b, n_top, n_bottom, markers)

    means_a = features[group_a].groupby(sample_key[group_a]).mean()[toplot]
    means_b = features[group_b].groupby(sample_key[group_b]).mean()[toplot]

    ma = means_a.copy(); ma['_group'] = labels[0]
    mb = means_b.copy(); mb['_group'] = labels[1]
    df = pd.concat([ma, mb]).melt(
        id_vars='_group', value_vars=toplot, var_name='feature', value_name='value')

    sns.swarmplot(data=df, x='feature', y='value', hue='_group',
                  dodge=True, order=toplot, ax=ax, **kwargs)

    if connect:
        dodge = 0.2
        common = means_a.index.intersection(means_b.index)
        for fi, feat in enumerate(toplot):
            for sid in common:
                va = means_a.loc[sid, feat] if sid in means_a.index else np.nan
                vb = means_b.loc[sid, feat] if sid in means_b.index else np.nan
                if np.isnan(va) or np.isnan(vb):
                    continue
                ax.plot([fi - dodge, fi + dodge], [va, vb],
                        color='gray', alpha=0.4, lw=0.8, zorder=0)

    if show:
        plt.show()
