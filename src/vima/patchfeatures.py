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
    """Return per-patch cell type counts.

    Args:
        cells: DataFrame with one row per cell.
        patch_meta: DataFrame with one row per patch (e.g. P.meta, F.obs, D.obs).
        sid_col: Column in `cells` giving each cell's sample ID.
        celltype_col: Column in `cells` giving each cell's type (cast to str).
        x_col: Column in `cells` giving each cell's x coordinate.
        y_col: Column in `cells` giving each cell's y coordinate.
        normalized: Whether to normalize counts by number of cells and add a totalcells column (default True).
        patch_sid_col: Column in `patch_meta` for sample ID (default 'sid').
        patch_x_col: Column in `patch_meta` for patch origin x (default 'x').
        patch_y_col: Column in `patch_meta` for patch origin y (default 'y').
        patch_size_col: Column in `patch_meta` for patch size (default 'patchsize').

    Returns:
        DataFrame indexed like `patch_meta`, columns = cell types, values = counts.
        Cells that overlap multiple patches (due to stride < patchsize) are counted
        in each overlapping patch.
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
    """Return per-patch mean marker expression profiles.

    Reads one .nc file at a time to avoid holding multiple samples in memory.

    Args:
        directory: Directory containing one {sid}.nc file per sample.
        patch_meta: DataFrame with one row per patch (e.g. P.meta, F.obs, D.obs).
        patch_sid_col: Column in patch_meta for sample ID (default 'sid').
        patch_x_col: Column in patch_meta for patch origin x in pixels (default 'x').
        patch_y_col: Column in patch_meta for patch origin y in pixels (default 'y').
        patch_size_in_pixels_col: Column in patch_meta for patch size in pixels (default 'patchsize').

    Returns:
        DataFrame indexed like patch_meta, columns = marker names,
        values = mean expression per patch.
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
    """Compare feature distributions between two patch groups.

    Uses mean rank difference (Wilcoxon/AUC equivalent) as the test statistic.
    By default, significance is assessed via donor-level permutation (group labels
    are flipped at the unit_of_analysis level). Pass Ttest=True to instead run a
    paired T-test on per-unit mean rank differences.

    Args:
        features: DataFrame (n_patches × n_features), e.g. from cell_type_counts or
                  expression_profiles.
        group_a: boolean array, length n_patches — first group (e.g. associated patches).
        group_b: boolean array or None — second group; defaults to ~group_a.
        unit_of_analysis: array-like of donor/sample IDs aligned with features rows.
                  Permutations (or T-test pairing) operate at the level of unique values.
        n_perms: number of permutations (default 100000); ignored when Ttest=True.
        seed: random seed for reproducibility; ignored when Ttest=True.
        corr_method: multiple-testing correction — 'benjamini-hochberg' (default) or 'bonferroni'.
        Ttest: if True, use a one-sample two-sided T-test on per-unit mean rank differences
               instead of permutation testing (default False).

    Returns:
        DataFrame indexed by feature name, columns: median_a, median_b, diff, pvals,
        pvals_adj. diff = median_a - median_b. Sorted by pvals ascending.
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