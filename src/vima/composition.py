import numpy as np
import pandas as pd


def cell_type_counts(
    cells,
    patch_meta,
    sid_col,
    celltype_col,
    x_col,
    y_col,
    patch_sid_col='sid',
    patch_x_col='x_microns',
    patch_y_col='y_microns',
    patch_size_col='ps',
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

    for sid, sid_cells in cells.groupby(sid_col):
        sid_patches = patch_meta[patch_meta[patch_sid_col] == sid]
        if len(sid_patches) == 0 or len(sid_cells) == 0:
            continue

        px = sid_patches[patch_x_col].values        # (n_patches,)
        py = sid_patches[patch_y_col].values
        ps = sid_patches[patch_size_col].values

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
