import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

def cell_type_counts(
    cells,
    patch_meta,
    sid_col,
    celltype_col,
    x_col,
    y_col,
    patch_sid_col='sid',
    patch_x_microns_col='x_microns',
    patch_y_microns_col='y_microns',
    patch_size_in_pixels_col='patchsize',
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

    for sid, sid_cells in pb(cells.groupby(sid_col)):
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

    return counts


def expression_profiles(
    directory,
    patch_meta,
    patch_sid_col='sid',
    patch_x_col='x',
    patch_y_col='y',
    patch_size_in_pixels_col='patchsize',
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

    for sid in pb(sids):
        sid_patches = patch_meta[patch_meta[patch_sid_col] == sid]
        sample = xr.open_dataarray(os.path.join(directory, f'{sid}.nc')).load()
        marker_names = sample.coords['marker'].values.tolist()
        data = sample.values  # (n_y, n_x, n_markers)
        del sample

        if result is None:
            result = pd.DataFrame(np.nan, index=patch_meta.index, columns=marker_names)

        xs  = sid_patches[patch_x_col].values.astype(int)
        ys  = sid_patches[patch_y_col].values.astype(int)
        pss = sid_patches[patch_size_in_pixels_col].values.astype(int)

        means = np.array([
            data[y:y+ps, x:x+ps, :].mean(axis=(0, 1))
            for x, y, ps in zip(xs, ys, pss)
        ])

        result.loc[sid_patches.index] = means
        del data

    return result


def diff_features(
    features,
    group_a,
    group_b=None,
    *,
    perm_key,
    method='mean_of_ranks',
    n_perms=100000,
    seed=None,
):
    """Compare feature distributions between two patch groups via donor-level permutation.

    Group labels are flipped at the donor level so the null distribution respects
    within-donor correlations.

    Args:
        features: DataFrame (n_patches × n_features), e.g. from cell_type_counts or
                  expression_profiles.
        group_a: boolean array, length n_patches — first group (e.g. associated patches).
        group_b: boolean array or None — second group; defaults to ~group_a.
        perm_key: array-like of donor/sample IDs aligned with features rows. Permutations
                  flip labels at the level of unique values in perm_key.
        method: test statistic to use —
                'mean'             global mean difference,
                'mean_of_ranks'    mean rank difference (Wilcoxon/AUC equivalent, default),
                'mean_of_medians'  mean of per-donor median differences; only donors
                                   with patches in both groups contribute.
        n_perms: number of permutations (default 100000).
        seed: random seed for reproducibility.

    Returns:
        DataFrame indexed by feature name, columns: median_a, median_b, diff, pval,
        pval_bonf. diff = median_a - median_b. Sorted by diff descending.
    """
    if method not in ('mean', 'mean_of_medians', 'mean_of_ranks'):
        raise ValueError(
            f'method must be "mean", "mean_of_medians", or "mean_of_ranks"; got {method!r}'
        )

    rng     = np.random.default_rng(seed)
    group_a = np.asarray(group_a, dtype=bool)
    group_b = ~group_a if group_b is None else np.asarray(group_b, dtype=bool)
    donors  = np.asarray(perm_key)

    X = features.values.astype(float)

    # Raw means always used for output columns, regardless of method
    median_a_raw = np.median(X[group_a], axis=0)
    median_b_raw = np.median(X[group_b], axis=0)

    unique_donors = np.unique(donors)
    n_donors      = len(unique_donors)

    if method == 'mean_of_ranks':
        X = rankdata(X, axis=0, nan_policy='raise')

    sum_a   = X[group_a].sum(axis=0)
    sum_b   = X[group_b].sum(axis=0)
    count_a = float(group_a.sum())
    count_b = float(group_b.sum())
    obs_diff = sum_a / count_a - sum_b / count_b

    da_sum   = np.zeros((n_donors, X.shape[1]))
    db_sum   = np.zeros((n_donors, X.shape[1]))
    da_count = np.zeros(n_donors)
    db_count = np.zeros(n_donors)
    for i, d in enumerate(unique_donors):
        in_d        = donors == d
        da_sum[i]   = X[in_d & group_a].sum(axis=0)
        db_sum[i]   = X[in_d & group_b].sum(axis=0)
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

    pval      = ((np.abs(null_diff) >= np.abs(obs_diff)).sum(axis=0) + 1) / (n_perms + 1)
    pval_bonf = np.minimum(pval * X.shape[1], 1.0)

    return pd.DataFrame({
        'median_a': median_a_raw,
        'median_b': median_b_raw,
        'diff':     median_a_raw - median_b_raw,
        'pval':     pval,
        'pval_bonf': pval_bonf,
    }, index=features.columns).sort_values('diff', ascending=False)