import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import xarray as xr
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
pb = lambda x, d: tqdm(x, ncols=100, desc=d)

_PALETTE = [
    [1, 0, 0],   # red
    [0, 1, 0],   # green
    [0, 0, 1],   # blue
    [0, 1, 1],   # cyan
    [1, 0, 1],   # magenta
    [1, 1, 0],   # yellow
]


# ── shared renderers ──────────────────────────────────────────────────────────

def _plot_separate(patches, markers, vmin, vmax, cmap='seismic', show=True):
    N, K = patches.shape[0], len(markers)
    fig, axes = plt.subplots(K, N, figsize=(N, K * 1.2), squeeze=False)
    for k, marker in enumerate(markers):
        for i in range(N):
            axes[k, i].imshow(patches[i, :, :, k], vmin=vmin[k], vmax=vmax[k], cmap=cmap)
            axes[k, i].axis('off')
        axes[k, 0].set_ylabel(marker, fontsize=9)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _plot_composite(patches, markers, colors, vmin, vmax, features=None, nx=5, ny=5, show=True,
                    subfig=None):
    """
    Render patches as additive-RGB composites on an nx × ny grid.

    Each marker is scaled to its ``(vmin, vmax)`` range and tinted by its color,
    and the tinted channels are summed into one RGB image per patch. When
    `features` are given (and there are enough patches), patches are placed on the
    grid by a 2D UMAP of `features` so that similar patches sit near each other.

    Parameters
    ----------
    patches
        Patch-by-y-by-x-by-marker pixel array.
    colors
        Per-marker ``[R, G, B]`` color.
    vmin, vmax
        Per-marker scaling bounds.
    features
        Optional per-patch vectors used to arrange patches on the grid.
    nx, ny
        Grid columns and rows.
    """
    N, ps, K = patches.shape[0], patches.shape[1], len(markers)

    rgb = np.zeros((N, ps, ps, 3))
    for k in range(K):
        scale = max(vmax[k] - vmin[k], 1e-8)
        scaled = np.clip((patches[:, :, :, k] - vmin[k]) / scale, 0, 1)
        rgb += scaled[:, :, :, None] * colors[k][None, None, None, :]
    rgb = np.clip(rgb, 0, 1)

    if features is not None and N >= 3:
        adata = sc.AnnData(X=np.array(features, dtype=float))
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=min(15, N - 1))
        sc.tl.umap(adata)
        coords = adata.obsm['X_umap'].copy()
        coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / max(coords[:, 0].max() - coords[:, 0].min(), 1e-8) * (nx - 1)
        coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) / max(coords[:, 1].max() - coords[:, 1].min(), 1e-8) * (ny - 1)
        gridpoints = np.array([[c, r] for r in range(ny) for c in range(nx)], dtype=float)
        cost = np.linalg.norm(gridpoints[:, None, :] - coords[None, :, :], axis=2)
        cell_inds, patch_inds = linear_sum_assignment(cost)
        cell_to_patch = dict(zip(cell_inds, patch_inds))
    else:
        cell_to_patch = {i: i for i in range(N)}

    fig = subfig if subfig is not None else plt.figure(figsize=(nx, ny))
    axs = fig.subplots(ny, nx)
    for ax in axs.flatten():
        ax.axis('off')
    for cell_i, patch_i in cell_to_patch.items():
        col, row = cell_i % nx, cell_i // nx
        axs[row, col].imshow(rgb[patch_i])

    legend_handles = [mpatches.Patch(facecolor=colors[k], label=markers[k]) for k in range(K)]
    fig.legend(handles=legend_handles, loc='lower center', ncol=K, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)
    if subfig is None:
        plt.tight_layout(rect=[0, 0.08, 1, 1])
    if show:
        plt.show()
    return fig


# ── MarkersInSpace ────────────────────────────────────────────────────────────

class MarkersInSpace:
    """
    Lazy per-sample marker cache backing the patch-display functions.

    Loads and caches only the requested markers per sample and computes shared
    color limits across all loaded samples, so repeated `show_separate` /
    `show_composite` calls stay fast and consistently scaled. The standalone
    `show_patches_separate` / `show_patches_composite` wrappers manage an
    instance for you; construct one directly only for fine-grained control.

    Parameters
    ----------
    directory
        Directory of ``<sid>.nc`` files.
    markers
        Markers to pre-register; otherwise loaded on first use.
    samples
        Optional ``{sid: DataArray}`` to read from instead of disk.
    percentile
        ``(low, high)`` percentiles defining the per-marker color limits.
    """

    def __init__(self, directory, markers=[], samples=None, percentile=(2, 98)):
        self.directory = directory
        self.samples = samples
        self.percentile = percentile
        self.markers = []
        self._marker_to_idx = {}
        self._arrays = {}   # {sid: np.ndarray(H, W, K)}
        self.vmin = {}      # {marker: float}
        self.vmax = {}      # {marker: float}
        if samples is not None:
            self._all_sids = set(samples.keys())
        else:
            self._all_sids = {f[:-3] for f in os.listdir(directory) if f.endswith('.nc')}
        if markers:
            self.add_markers(markers)

    # ── internal loading ──────────────────────────────────────────────────────

    def _read_sid(self, sid, markers):
        """Read (H, W, K) array for the given sid and markers from disk or cache."""
        if (self.samples is not None
                and sid in self.samples
                and set(markers).issubset(self.samples[sid].coords['marker'].values)):
            return np.array(self.samples[sid].sel(marker=markers).values, dtype=np.float32)
        return np.array(
            xr.open_dataarray(f'{self.directory}/{sid}.nc').sel(marker=markers).values,
            dtype=np.float32)

    def _update_stats(self):
        """Recompute vmin/vmax over all non-empty pixels across all loaded samples."""
        if not self._arrays or not self.markers:
            return
        chunks = []
        for arr in self._arrays.values():
            mask = (arr != 0).any(axis=-1)  # (H, W) bool
            chunks.append(arr[mask])         # (n_nonzero, K)
        all_pixels = np.concatenate(chunks, axis=0)  # (total_nonzero, K)
        p_lo, p_hi = np.percentile(all_pixels, [self.percentile[0], self.percentile[1]], axis=0)
        for k, marker in enumerate(self.markers):
            self.vmin[marker] = float(p_lo[k])
            self.vmax[marker] = float(p_hi[k])

    # ── public API ────────────────────────────────────────────────────────────

    def add_markers(self, new_markers):
        """Register and cache new markers for all already-loaded samples."""
        new_markers = [m for m in new_markers if m not in self._marker_to_idx]
        if not new_markers:
            return
        for i, m in enumerate(new_markers):
            self._marker_to_idx[m] = len(self.markers) + i
        self.markers.extend(new_markers)
        for sid, arr in pb(self._arrays.items(), f'Adding {len(new_markers)} markers'):
            new_data = self._read_sid(sid, new_markers)
            self._arrays[sid] = np.concatenate([arr, new_data], axis=-1)
        self._ensure_sids_loaded(self._all_sids)  # load remaining sids for dataset-wide stats
        self._update_stats()

    def _ensure_sids_loaded(self, sids):
        added = False
        for sid in pb(sids, 'Reading samples'):
            if sid not in self._arrays and self.markers:
                self._arrays[sid] = self._read_sid(sid, self.markers)
                added = True
        if added:
            self._update_stats()

    def _extract_patches(self, patchmeta, marker_indices):
        ps = int(patchmeta['patchsize'].iloc[0])
        result = np.empty((len(patchmeta), ps, ps, len(marker_indices)), dtype=np.float32)
        for i, (_, row) in enumerate(patchmeta.iterrows()):
            result[i] = self._arrays[row.sid][int(row.y):int(row.y)+ps,
                                              int(row.x):int(row.x)+ps, :][:, :, marker_indices]
        return result

    def _resolve_scale(self, markers, vmin, vmax):
        K = len(markers)
        if vmin is None:
            vmin = [self.vmin[m] for m in markers]
        elif not hasattr(vmin, '__len__'):
            vmin = [float(vmin)] * K
        if vmax is None:
            vmax = [self.vmax[m] for m in markers]
        elif not hasattr(vmax, '__len__'):
            vmax = [float(vmax)] * K
        return list(vmin), list(vmax)

    # ── plotting ──────────────────────────────────────────────────────────────

    def show_separate(self, patchmeta, markers=None, n=25, seed=None,
                      cmap='seismic', vmin=None, vmax=None, show=True):
        """
        Show patches in a grid: one row per marker, one column per patch.

        Parameters
        ----------
        patchmeta
            Patches to draw from.
        markers
            Markers to display; default all cached (new ones auto-loaded).
        n
            Number of patches to show (randomly downsampled if more).
        vmin, vmax
            Per-marker color bounds (scalar, list, or None for dataset-wide).
        """
        if markers is None:
            markers = list(self.markers)
        self.add_markers([m for m in markers if m not in self._marker_to_idx])
        self._ensure_sids_loaded(patchmeta.sid.unique())

        if len(patchmeta) > n:
            patchmeta = patchmeta.sample(n=n, random_state=seed)

        marker_indices = [self._marker_to_idx[m] for m in markers]
        patches = self._extract_patches(patchmeta, marker_indices)
        vmin, vmax = self._resolve_scale(markers, vmin, vmax)
        return _plot_separate(patches, markers, vmin, vmax, cmap, show=show)

    def show_composite(self, patchmeta, markers=None, features=None, colors=None,
                       n=25, nx=5, ny=5, seed=None, vmin=None, vmax=None, show=True, subfig=None):
        """
        Show patches as additive RGB composites in an nx × ny grid.

        Parameters
        ----------
        patchmeta
            Patches to draw from.
        markers
            Markers to display; default all cached (new ones auto-loaded).
        features
            Per-patch vectors aligned with `patchmeta` rows; when given, patches
            are arranged on the grid by 2D UMAP so similar patches sit together.
        colors
            Per-marker ``[R, G, B]`` colors; auto-assigned from a palette if None.
        n
            Number of patches (capped at ``nx*ny``, randomly downsampled if more).
        nx, ny
            Grid columns and rows.
        vmin, vmax
            Per-marker color bounds (scalar, list, or None for dataset-wide).
        """
        n = min(n, nx * ny)
        rng = np.random.default_rng(seed)

        if markers is None:
            markers = list(self.markers)
        self.add_markers([m for m in markers if m not in self._marker_to_idx])
        self._ensure_sids_loaded(patchmeta.sid.unique())

        if len(patchmeta) > n:
            positions = rng.choice(len(patchmeta), size=n, replace=False)
            patchmeta = patchmeta.iloc[positions]
            if features is not None:
                features = np.array(features)[positions]

        K = len(markers)
        if colors is None:
            if K > len(_PALETTE):
                warnings.warn(f'{K} markers but palette only has {len(_PALETTE)} colors; truncating')
                markers = markers[:len(_PALETTE)]
                K = len(markers)
            colors = [np.array(c, dtype=float) for c in _PALETTE[:K]]
        else:
            colors = [np.array(c, dtype=float) for c in colors]

        marker_indices = [self._marker_to_idx[m] for m in markers]
        patches = self._extract_patches(patchmeta, marker_indices)
        vmin, vmax = self._resolve_scale(markers, vmin, vmax)
        return _plot_composite(patches, markers, colors, vmin, vmax, features, nx, ny, show=show,
                               subfig=subfig)


# ── standalone convenience functions (backed by a global MarkersInSpace) ─────

default_mis = None  # accessible as vima.vis.default_mis; reset if directory changes


def _get_default_mis(directory, samples):
    global default_mis
    if default_mis is None or default_mis.directory != directory:
        default_mis = MarkersInSpace(directory, samples=samples)
    return default_mis


def show_patches_separate(patchmeta, markers, directory, samples=None,
                          n=25, seed=None, cmap='seismic', vmin=None, vmax=None, show=True):
    """
    Show patches by marker, in a grid of one row per marker.

    Reads pixel data from ``directory`` via a cached `MarkersInSpace` (reused
    across calls with the same directory for speed and consistent color scales).
    See `MarkersInSpace.show_separate` for the display parameters.

    Parameters
    ----------
    patchmeta
        Patches to draw from (e.g. a subset of ``D.obs``).
    markers
        Markers to display.
    directory
        Directory of ``<sid>.nc`` pixel files.
    samples
        Optional ``{sid: DataArray}`` to read from instead of disk.
    """
    return _get_default_mis(directory, samples).show_separate(
        patchmeta, markers, n=n, seed=seed, cmap=cmap, vmin=vmin, vmax=vmax, show=show)


def show_patches_composite(patchmeta, markers, directory, samples=None,
                            features=None, colors=None,
                            n=25, nx=5, ny=5, seed=None, vmin=None, vmax=None, show=True,
                            subfig=None):
    """
    Show patches as additive RGB composites in an nx × ny grid.

    Reads pixel data from ``directory`` via a cached `MarkersInSpace` (reused
    across calls with the same directory for speed and consistent color scales).
    See `MarkersInSpace.show_composite` for the display parameters, including
    `features`, which arranges similar patches near each other on the grid.

    Parameters
    ----------
    patchmeta
        Patches to draw from (e.g. a subset of ``D.obs``).
    markers
        Markers to composite (each assigned a color).
    directory
        Directory of ``<sid>.nc`` pixel files.
    samples
        Optional ``{sid: DataArray}`` to read from instead of disk.
    """
    return _get_default_mis(directory, samples).show_composite(
        patchmeta, markers, features=features, colors=colors,
        n=n, nx=nx, ny=ny, seed=seed, vmin=vmin, vmax=vmax, show=show, subfig=subfig)


def show_patches_cells(patchmeta, cells, x_col, y_col, celltype_col,
                       pixelsize_microns, nx=5, ny=5, sid_col='sid',
                       colors=None, seed=None, s=8, show=True, subfig=None,
                       include_only=None):
    """
    Show a grid of random patches with their segmented cells as colored dots.

    Parameters
    ----------
    patchmeta
        Patches to draw from, with 'sid', 'x_microns', 'y_microns', 'patchsize'.
    cells
        Segmented cells with sample ID, coordinates, and type.
    x_col, y_col
        Cell coordinate columns, in the same units as the patch micron columns.
    celltype_col
        Cell-type label column.
    pixelsize_microns
        Micron size of one pixel, used to size each patch.
    nx, ny
        Grid columns and rows (``nx*ny`` patches shown).
    colors
        ``{cell_type: color}`` for explicit colors; auto-assigned if None.
    include_only
        If given, only these cell types are drawn.
    """
    n = nx * ny

    # Downsample patches
    if len(patchmeta) > n:
        patchmeta = patchmeta.sample(n=n, random_state=seed)
    patchmeta = patchmeta.reset_index(drop=True)

    extent = int(patchmeta['patchsize'].iloc[0]) * pixelsize_microns

    if include_only is not None:
        cells = cells[cells[celltype_col].isin(include_only)]

    # Build color map for cell types
    all_types = sorted(cells[celltype_col].dropna().unique())
    if colors is None:
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(all_types), 1)))
        colors = {ct: palette[i] for i, ct in enumerate(all_types)}

    fig = subfig if subfig is not None else plt.figure(figsize=(nx * 2, ny * 2))
    axs = fig.subplots(ny, nx, squeeze=False)
    for ax in axs.flatten():
        ax.set_visible(False)

    for plot_idx, (_, row) in enumerate(patchmeta.iterrows()):
        ax = axs[plot_idx // nx, plot_idx % nx]
        ax.set_visible(True)

        x0, y0 = row.x_microns, row.y_microns
        x1, y1 = x0 + extent, y0 + extent

        patch_cells = cells[
            (cells[sid_col] == row['sid']) &
            (cells[x_col] >= x0) & (cells[x_col] < x1) &
            (cells[y_col] >= y0) & (cells[y_col] < y1)
        ]

        ax.set_facecolor('#f0f0f0')
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect('equal')
        ax.axis('off')

        for ct, grp in patch_cells.groupby(celltype_col, observed=True):
            ax.scatter(grp[x_col], grp[y_col],
                       c=[colors.get(ct, 'gray')], s=s, linewidths=0)

    handles = [mpatches.Patch(facecolor=colors.get(ct, 'gray'), label=ct)
               for ct in all_types]
    ncol_legend = min(len(all_types), 6)
    bottom_margin = 0.04 * int(np.ceil(len(all_types) / ncol_legend)) + 0.02
    fig.legend(handles=handles, loc='lower center', ncol=ncol_legend,
               frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)
    if subfig is None:
        plt.tight_layout(rect=[0, bottom_margin, 1, 1])

    if show:
        plt.show()
    return fig
