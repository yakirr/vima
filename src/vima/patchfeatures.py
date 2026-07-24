import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata
from tqdm import tqdm
pb = lambda x, desc: tqdm(x, ncols=100, desc=desc)

def cell_type_counts(
    cells,
    patch_meta,
    sid_col,
    celltype_col,
    x_col,
    y_col,
    normalized=True,
    patch_sid_col='sid',
    patch_x_microns_col='x_microns',
    patch_y_microns_col='y_microns',
    patch_size_in_pixels_col='patchsize',
    include_totalcells=False,
    pixel_size_microns=10
):
    """
    Compute per-patch cell type counts by spatial overlap.

    Assigns each cell to every patch whose extent contains it, so with
    overlapping patches (stride < patchsize) a cell contributes to several
    patches.

    Parameters
    ----------
    cells
        One row per cell, with sample ID, type, and coordinates.
    patch_meta
        One row per patch (e.g. ``P.meta``, ``F.obs``, ``D.obs``).
    sid_col, celltype_col, x_col, y_col
        Columns in `cells` for sample ID, cell type (cast to str), and x/y
        coordinates in microns.
    normalized
        Divide each patch's counts by its total cell count (default True).
    include_totalcells
        Add a 'totalcells' column of raw per-patch cell counts.
    pixel_size_microns
        Micron size of one pixel, used to convert patch size to microns.

    Returns
    -------
    DataFrame
        Indexed like `patch_meta`, one column per cell type.
    """
    cells = cells.copy()
    cells[celltype_col] = cells[celltype_col].astype(str)

    cell_types = sorted(cells[celltype_col].unique())
    counts = pd.DataFrame(0, index=patch_meta.index, columns=cell_types, dtype=int)

    for sid, sid_cells in pb(cells.groupby(sid_col), 'cell_type_counts'):
        sid_patches = patch_meta[patch_meta[patch_sid_col] == sid]
        if len(sid_patches) == 0 or len(sid_cells) == 0:
            continue

        px = sid_patches[patch_x_microns_col].values        # (n_patches,)
        py = sid_patches[patch_y_microns_col].values
        ps = sid_patches[patch_size_in_pixels_col].values

        cx = sid_cells[x_col].values                # (n_cells,)
        cy = sid_cells[y_col].values
        ct = sid_cells[celltype_col].values

        # in_patch[i, j] is True when cell j falls inside patch i
        in_x = (px[:, None] <= cx[None, :]) & (cx[None, :] < px[:, None] + pixel_size_microns*ps[:, None])
        in_y = (py[:, None] <= cy[None, :]) & (cy[None, :] < py[:, None] + pixel_size_microns*ps[:, None])
        in_patch = in_x & in_y                      # (n_patches, n_cells)

        for ct_val in cell_types:
            counts.loc[sid_patches.index, ct_val] += in_patch[:, ct == ct_val].sum(axis=1)

    if normalized:
        totals = counts.sum(axis=1)
        counts = counts.div(totals, axis=0).fillna(0)
    if include_totalcells:
        counts['totalcells'] = totals

    return counts


def expression_profiles(
    directory,
    patch_meta,
    patch_sid_col='sid',
    patch_x_col='x',
    patch_y_col='y',
    patch_size_in_pixels_col='patchsize',
    per_nonempty_pixel=True,
):
    """
    Compute per-patch mean marker expression profiles.

    Parameters
    ----------
    directory
        Directory of one ``{sid}.nc`` file per sample (e.g. the normalized
        pixel matrices).
    patch_meta
        One row per patch (e.g. ``P.meta``, ``F.obs``, ``D.obs``).
    per_nonempty_pixel
        Average over non-empty pixels only (default True); otherwise average
        over all pixels in the patch.

    Returns
    -------
    DataFrame
        Indexed like `patch_meta`, one column per marker.
    """
    sids = patch_meta[patch_sid_col].unique()
    for sid in sids:
        path = os.path.join(directory, f'{sid}.nc')
        if not os.path.exists(path):
            raise FileNotFoundError(f'No .nc file for sample {sid!r} in {directory}')

    result = None

    for sid in pb(sids, 'expression_profiles'):
        sid_patches = patch_meta[patch_meta[patch_sid_col] == sid]
        sample = xr.open_dataarray(os.path.join(directory, f'{sid}.nc')).load()
        marker_names = sample.coords['marker'].values.tolist()
        data = sample.values  # (n_y, n_x, n_markers)
        if per_nonempty_pixel:
            mask = ~(data == 0).all(axis=-1)  # (n_y, n_x)
        else:
            mask = np.ones(data.shape[:2], dtype=bool)
        del sample

        if result is None:
            result = pd.DataFrame(np.nan, index=patch_meta.index, columns=marker_names)

        xs  = sid_patches[patch_x_col].values.astype(int)
        ys  = sid_patches[patch_y_col].values.astype(int)
        pss = sid_patches[patch_size_in_pixels_col].values.astype(int)

        sums = np.array([
            data[y:y+ps, x:x+ps, :].sum(axis=(0, 1))
            for x, y, ps in zip(xs, ys, pss)
        ])
        npixels = np.array([
            mask[y:y+ps, x:x+ps].sum()
            for x, y, ps in zip(xs, ys, pss)
        ])

        result.loc[sid_patches.index] = sums / npixels[:, None]
        del data

    return result


def _permutation_pvals(X_ranked, group_a, group_b, donors, n_perms, rng):
    """
    Compute two-sided permutation p-values for the between-group rank difference.

    The null is generated by flipping each donor's group assignment as a whole,
    so the test respects donor-level (not patch-level) exchangeability.

    Parameters
    ----------
    X_ranked
        Patch-by-feature rank matrix.
    group_a, group_b
        Boolean masks over patches for the two groups.
    donors
        Per-patch donor/sample label whose assignment is flipped as a unit.

    Returns
    -------
    ndarray
        One p-value per feature.
    """
    sum_a    = X_ranked[group_a].sum(axis=0)
    sum_b    = X_ranked[group_b].sum(axis=0)
    count_a  = float(group_a.sum())
    count_b  = float(group_b.sum())
    obs_diff = sum_a / count_a - sum_b / count_b

    unique_donors = np.unique(donors)
    n_donors      = len(unique_donors)
    da_sum   = np.zeros((n_donors, X_ranked.shape[1]))
    db_sum   = np.zeros((n_donors, X_ranked.shape[1]))
    da_count = np.zeros(n_donors)
    db_count = np.zeros(n_donors)
    for i, d in enumerate(unique_donors):
        in_d        = donors == d
        da_sum[i]   = X_ranked[in_d & group_a].sum(axis=0)
        db_sum[i]   = X_ranked[in_d & group_b].sum(axis=0)
        da_count[i] = (in_d & group_a).sum()
        db_count[i] = (in_d & group_b).sum()

    delta_sum   = db_sum - da_sum
    delta_count = db_count - da_count
    flip = (rng.random((n_perms, n_donors)) < 0.5).astype(float)

    sum_a_null   = sum_a   + flip @ delta_sum
    count_a_null = count_a + flip @ delta_count
    sum_b_null   = (sum_a + sum_b) - sum_a_null
    count_b_null = (count_a + count_b) - count_a_null
    count_a_null = np.maximum(count_a_null, 1.0)
    count_b_null = np.maximum(count_b_null, 1.0)
    null_diff = sum_a_null / count_a_null[:, None] - sum_b_null / count_b_null[:, None]

    return ((np.abs(null_diff) >= np.abs(obs_diff)).sum(axis=0) + 1) / (n_perms + 1)


def _ttest_pvals(X_ranked, group_a, group_b, donors):
    from scipy import stats
    global_mean_a = X_ranked[group_a].mean(axis=0)
    global_mean_b = X_ranked[group_b].mean(axis=0)
    diffs = []
    for d in np.unique(donors):
        in_d   = donors == d
        a_rows = X_ranked[in_d & group_a]
        b_rows = X_ranked[in_d & group_b]
        mean_a = a_rows.mean(axis=0) if len(a_rows) > 0 else global_mean_a
        mean_b = b_rows.mean(axis=0) if len(b_rows) > 0 else global_mean_b
        diffs.append(mean_a - mean_b)
    if len(diffs) < 2:
        raise ValueError(f'T-test requires at least 2 units; got {len(diffs)}')
    _, pvals = stats.ttest_1samp(np.array(diffs), 0, axis=0)
    return pvals


def test_features(
    features,
    group_a,
    group_b=None,
    *,
    unit_of_analysis,
    n_perms=100000,
    seed=None,
    corr_method='benjamini-hochberg',
    Ttest=False,
):
    """
    Compare feature distributions between two patch groups.

    The test statistic is the mean rank difference between groups (equivalent to
    Wilcoxon/AUC). Significance is assessed by permutation at the
    `unit_of_analysis` level (group labels are flipped per unit), or by a paired
    T-test on per-unit mean rank differences when ``Ttest=True``. The T-test is
    useful in small datasets where permutation is underpowered but magnitude
    differences between groups are large.

    Parameters
    ----------
    features
        Patch-by-feature values, e.g. from `cell_type_counts` or
        `expression_profiles`. Must be a DataFrame indexed by patch.
    group_a
        Boolean Series over patches selecting the first group.
    group_b
        Boolean Series selecting the second group; defaults to ``~group_a``.
    unit_of_analysis
        Series mapping each patch to its donor/sample; permutation and T-test
        pairing operate over these units.
    n_perms
        Number of permutations (ignored when ``Ttest=True``).
    corr_method
        Multiple-testing correction: 'benjamini-hochberg' or 'bonferroni'.
    Ttest
        Use a two-sided one-sample T-test instead of permutation.

    Returns
    -------
    DataFrame
        Indexed by feature, sorted by ascending p-value, with medians per group,
        ``diff_median``, ``pvals``, and ``pvals_adj`` (and per-group means and
        log2 fold changes when all features are non-negative).
    """
    if corr_method not in ('benjamini-hochberg', 'bonferroni'):
        raise ValueError(
            f'corr_method must be "benjamini-hochberg" or "bonferroni"; got {corr_method!r}'
        )

    rng = np.random.default_rng(seed)

    def _align(s, name):
        if not isinstance(s, pd.Series):
            raise TypeError(f'{name} must be a pandas Series')
        missing = features.index.difference(s.index)
        extra   = s.index.difference(features.index)
        if len(missing) or len(extra):
            raise ValueError(
                f'{name} index does not match features.index: '
                f'{len(missing)} missing, {len(extra)} extra label(s)')
        return s.reindex(features.index).to_numpy()

    group_a  = _align(group_a, 'group_a').astype(bool)
    group_b  = ~group_a if group_b is None else _align(group_b, 'group_b').astype(bool)
    donors   = _align(unit_of_analysis, 'unit_of_analysis')
    X = features.values.astype(float)

    print(f'Comparing {group_a.sum()} patches (Group A) to {group_b.sum()} patches (Group B).')
    if group_a.sum() == 0 or group_b.sum() == 0:
        raise ValueError('Each group must have at least one patch')

    median_a_raw = np.median(X[group_a], axis=0)
    median_b_raw = np.median(X[group_b], axis=0)

    X = rankdata(X, axis=0, nan_policy='raise')

    if Ttest:
        pvals = _ttest_pvals(X, group_a, group_b, donors)
    else:
        pvals = _permutation_pvals(X, group_a, group_b, donors, n_perms, rng)

    if corr_method == 'benjamini-hochberg':
        from statsmodels.stats.multitest import multipletests
        _, pvals_adj, _, _ = multipletests(pvals, method='fdr_bh')
    else:
        pvals_adj = np.minimum(pvals * len(features.columns), 1.0)

    nonneg = (features.values >= 0).all()

    if nonneg:
        mean_a = features.values[group_a].mean(axis=0)
        mean_b = features.values[group_b].mean(axis=0)
        result = pd.DataFrame({
            'median_a':          median_a_raw,
            'median_b':          median_b_raw,
            'mean_a':      mean_a,
            'mean_b':      mean_b,
            'log2fc':            np.log2((median_a_raw + 1e-9) / (median_b_raw + 1e-9)),
            'log2fc_means': np.log2((mean_a + 1e-9) / (mean_b + 1e-9)),
            'diff_median':     median_a_raw - median_b_raw,
        }, index=features.columns)
    else:
        result = pd.DataFrame({
            'median_a': median_a_raw,
            'median_b': median_b_raw,
            'diff_median':     median_a_raw - median_b_raw,
        }, index=features.columns)

    result['pvals']     = pvals
    result['pvals_adj'] = pvals_adj

    return result.sort_values('pvals', ascending=True)