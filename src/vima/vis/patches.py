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


def _plot_composite(patches, markers, colors, vmin, vmax, features=None, nx=5, ny=5, show=True):
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

    fig, axs = plt.subplots(ny, nx, figsize=(nx, ny))
    for ax in axs.flatten():
        ax.axis('off')
    for cell_i, patch_i in cell_to_patch.items():
        col, row = cell_i % nx, cell_i // nx
        axs[row, col].imshow(rgb[patch_i])

    legend_handles = [mpatches.Patch(facecolor=colors[k], label=markers[k]) for k in range(K)]
    fig.legend(handles=legend_handles, loc='lower center', ncol=K, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if show:
        plt.show()
    return fig


# ── MarkersInSpace ────────────────────────────────────────────────────────────

class MarkersInSpace:
    """Lazy cache of per-sample spatial arrays for a growing set of markers.

    Stores one (H, W, K) float32 array per sample for only the K registered
    markers, rather than storing extracted patches (which would duplicate each
    pixel ~16× at default stride/patchsize settings). New samples and markers
    are loaded on demand when show_separate / show_composite are called.
    vmin/vmax are computed over all non-empty pixels across all loaded samples
    and updated whenever new samples or markers are added.

    Args:
        directory: Directory containing <sid>.nc files.
        markers: Optional list of markers to pre-register (loaded lazily on first plot).
        samples: Optional dict {sid: xarray.DataArray} to use instead of disk reads.
        percentile: (low, high) percentile pair used to compute vmin/vmax (default (2, 98)).
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
        """Show patches in a grid: one row per marker, one column per patch.

        Args:
            patchmeta: DataFrame subset of patches to draw from.
            markers: Markers to display (defaults to all cached; new ones auto-loaded).
            n: Number of patches to show (randomly downsampled if patchmeta is larger).
            seed: Random seed for downsampling.
            cmap: Matplotlib colormap.
            vmin, vmax: Per-marker bounds (scalar, list, or None to use dataset-wide stats).
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
                       n=25, nx=5, ny=5, seed=None, vmin=None, vmax=None, show=True):
        """Show patches as additive RGB composites in an nx × ny grid.

        Args:
            patchmeta: DataFrame subset of patches to draw from.
            markers: Markers to display (defaults to all cached; new ones auto-loaded).
            features: Array shape (len(patchmeta), D) aligned with patchmeta rows.
                If provided, patches are arranged by 2D UMAP + Hungarian assignment.
            colors: Per-marker [R, G, B] lists (auto-assigned from palette if None).
            n: Number of patches (capped at nx*ny, randomly downsampled if needed).
            nx, ny: Grid dimensions (columns, rows).
            seed: Random seed for downsampling.
            vmin, vmax: Per-marker bounds (scalar, list, or None to use dataset-wide stats).
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
        return _plot_composite(patches, markers, colors, vmin, vmax, features, nx, ny, show=show)


# ── standalone convenience functions (backed by a global MarkersInSpace) ─────

default_mis = None  # accessible as vima.vis.default_mis; reset if directory changes


def _get_default_mis(directory, samples):
    global default_mis
    if default_mis is None or default_mis.directory != directory:
        default_mis = MarkersInSpace(directory, samples=samples)
    return default_mis


def show_patches_separate(patchmeta, markers, directory, samples=None,
                          n=25, seed=None, cmap='seismic', vmin=None, vmax=None, show=True):
    """Convenience wrapper around MarkersInSpace.show_separate using a global cache.

    On the first call (or when directory changes) a new MarkersInSpace instance is
    created and stored in vima.vis.default_mis. Subsequent calls with the same
    directory reuse it, giving fast repeated plots with consistent color scales.
    """
    return _get_default_mis(directory, samples).show_separate(
        patchmeta, markers, n=n, seed=seed, cmap=cmap, vmin=vmin, vmax=vmax, show=show)


def show_patches_composite(patchmeta, markers, directory, samples=None,
                            features=None, colors=None,
                            n=25, nx=5, ny=5, seed=None, vmin=None, vmax=None, show=True):
    """Convenience wrapper around MarkersInSpace.show_composite using a global cache.

    On the first call (or when directory changes) a new MarkersInSpace instance is
    created and stored in vima.vis.default_mis. Subsequent calls with the same
    directory reuse it, giving fast repeated plots with consistent color scales.
    """
    return _get_default_mis(directory, samples).show_composite(
        patchmeta, markers, features=features, colors=colors,
        n=n, nx=nx, ny=ny, seed=seed, vmin=vmin, vmax=vmax, show=show)
