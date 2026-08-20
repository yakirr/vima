import numpy as numpy
import numpy as np
import pandas as pd
import xarray as xr
import cv2 as cv2
import scanpy as sc
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import gc, os
from . import util, dimreduce
from .._settings import settings, logger, Verbosity

def _lisi_ratio(conn, labels):
    from scipy.sparse import issparse, diags as sp_diags
    labels = np.asarray(labels)
    unique, inv = np.unique(labels, return_inverse=True)
    n_cats = len(unique)
    if n_cats < 2:
        return float('nan')
    n = conn.shape[0]
    L = np.zeros((n, n_cats))
    L[np.arange(n), inv] = 1.0
    rs = np.asarray(conn.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    freq = (sp_diags(1.0 / rs) @ conn) @ L
    if issparse(freq):
        freq = freq.toarray()
    mean_lisi = (1.0 / (freq ** 2).sum(axis=1)).mean()
    global_q = np.bincount(inv).astype(float) / n
    return mean_lisi, 1.0 / (global_q ** 2).sum()

def visualize_pixels(pixels, ntoplot, input, colorby):
    """
    UMAP a random subset of pixels, colored by covariates, and report mixing.

    Embeds ``ntoplot`` randomly sampled pixels using their PC columns and plots
    the UMAP colored by each covariate in ``colorby``. For each covariate,
    prints an average LISI ratio quantifying how well its categories mix (the
    printed baseline marks perfect mixing, 1 marks no mixing).

    Parameters
    ----------
    input
        Label for the representation being embedded, used in plot titles.
    colorby
        Covariate column names to color the UMAP by.

    Returns
    -------
    AnnData
        The embedded subset, with the UMAP and neighbor graph attached.
    """
    pcs = [c for c in pixels.columns if c.startswith('PC')]
    metavars = [c for c in pixels.columns if c not in pcs]
    np.random.seed(0)
    ix = np.random.choice(len(pixels), replace=False, size=ntoplot)
    toplot = pixels.iloc[ix]
    obs = toplot[metavars].copy()
    obs.index = obs.index.astype(str)
    toplot_ad = ad.AnnData(X=toplot[pcs], obs=obs)
    sc.pp.neighbors(toplot_ad, use_rep='X')
    sc.tl.umap(toplot_ad)
    
    for metavar in colorby:
        if settings.show_plots('verbose'):
            sns.scatterplot(x='PC1', y='PC2', hue=metavar, data=toplot, palette='Set1', s=1, legend=False)
            plt.title(metavar)
            settings.show(f'pc1_pc2_by_{metavar}')
        sc.pl.umap(toplot_ad, color=metavar, legend_loc=None, frameon=False,
                   title=f'pixels UMAPed using {input}, colored by {metavar}', show=False)
        settings.show(f'pixel_umap_by_{metavar}')
        
        # print LISI ratio
        n_unique = toplot_ad.obs[metavar].nunique()
        if n_unique < 2 or n_unique > 1000:
            logger.info(f'Avg LISI ({metavar}): skipped ({n_unique} unique values)')
        else:
            lisi, baseline = _lisi_ratio(toplot_ad.obsp['connectivities'], toplot_ad.obs[metavar])
            logger.info(f'Avg LISI ({metavar}): {lisi:.2f}  [{baseline:.2f} = perfect mixing, 1 = not mixed]')

    return toplot_ad

def add_covs(pca, sid_to_covs):
    """
    Attach per-sample covariate columns to a pixel table (in place).

    Maps each covariate in ``sid_to_covs`` onto the pixel table via its 'sid'
    column.

    Parameters
    ----------
    sid_to_covs
        DataFrame indexed by sample ID with one column per covariate, or None.

    Returns
    -------
    list
        Covariate names including 'sid'.
    """
    if sid_to_covs is not None:
        cov_names = list(sid_to_covs.columns)
    else:
        cov_names = []
    for cov_name in cov_names:
        pca[cov_name] = pca['sid'].map(sid_to_covs[cov_name])
    return ['sid'] + cov_names

def collapse_markers(normeddir, markers, pseudomarker, outdir, masksdir=None):
    """
    Keep a subset of markers and fold all the others into one pseudomarker.

    Reads the normalized pixel matrices written by `prepare_merfish`,
    `prepare_xenium5k`, or `nonst.prepare` (for transcriptomic data the markers
    are genes), retains `markers`, and replaces every other marker with a single
    channel named `pseudomarker` holding their combined signal. Because the
    stored values are log-normalized, they are exponentiated, summed, and
    log-normalized again, so the pseudomarker is on the same scale as a real
    marker. The dataset-wide means and stds stored on each file are rewritten to
    match the new marker set: retained markers keep their existing values and the
    pseudomarker's are computed from the collapsed data.

    Parameters
    ----------
    normeddir
        Directory of normalized ``.nc`` files, e.g. ``{outdir}/normalized``.
    markers
        Markers to retain, in the order they should appear; the pseudomarker is
        appended after them. Markers absent from the data are skipped with a
        warning.
    pseudomarker
        Name for the new channel, e.g. ``'nonimmune'``.
    outdir
        Directory to write the collapsed ``.nc`` files to. This is a
        ``normalized``-style directory, so to build a dataset that
        `pca_pixels` can be pointed at, pass ``f'{newroot}/normalized'`` and copy
        the masks to ``f'{newroot}/masks'``.
    masksdir
        Directory of tissue masks, used to compute the pseudomarker's moments
        over non-empty pixels only. Defaults to ``masks`` alongside `normeddir`.
    """
    import netCDF4

    if masksdir is None:
        masksdir = os.path.join(os.path.dirname(os.path.normpath(normeddir)), 'masks')

    sids = [os.path.splitext(f)[0]
        for f in os.listdir(normeddir) if f.endswith('.nc') and not f.startswith('.')]
    if len(sids) == 0:
        logger.warning(f'No .nc files found in {normeddir}. Check your path and try again.')
        return
    os.makedirs(outdir, exist_ok=True)

    # decide on the new marker set, using the first sample as the reference
    da = xr.open_dataarray(f'{normeddir}/{sids[0]}.nc')
    ref_markers = list(da.marker.values)
    ref_means, ref_stds = da.attrs['means'], da.attrs['stds']
    da.close(); del da

    if pseudomarker in ref_markers:
        raise ValueError(f"'{pseudomarker}' is already a marker; pick a name for the "
                         f"pseudomarker that isn't in the data.")
    missing = [m for m in markers if m not in ref_markers]
    if missing:
        logger.warning(f'{len(missing)} of the {len(markers)} requested markers are not in the '
                       f'data and will be skipped: {missing}')
    keep = [m for m in markers if m in ref_markers]
    if len(keep) == 0:
        raise ValueError('None of the requested markers are in the data.')
    if len(keep) == len(ref_markers):
        logger.warning(f'All {len(ref_markers)} markers were retained, so {pseudomarker} will be '
                       f'empty.')
    logger.info(f'Keeping {len(keep)} markers and collapsing the other '
                f'{len(ref_markers) - len(keep)} into {pseudomarker}.')

    # collapse each sample, accumulating the pseudomarker's per-sample moments as we go
    sample_sids, sample_means, sample_stds, sample_npixels = [], [], [], []
    for sid in settings.progress(sids, name='collapsing markers'):
        da = xr.open_dataarray(f'{normeddir}/{sid}.nc')
        mask_da = xr.open_dataarray(f'{masksdir}/{sid}.nc')

        sid_markers = list(da.marker.values)
        if sid_markers != ref_markers:
            logger.warning(f'{sid} has different markers ({len(sid_markers)}) than {sids[0]} '
                           f'({len(ref_markers)}); matching by name.')
        absent = [m for m in keep if m not in sid_markers]
        if absent:
            raise ValueError(f'{sid} is missing {len(absent)} of the markers to retain: {absent}')
        ix = {m: i for i, m in enumerate(sid_markers)}
        keep_ix = [ix[m] for m in keep]
        keep_ix_set = set(keep_ix)
        drop_ix = [i for i in range(len(sid_markers)) if i not in keep_ix_set]

        # load raw arrays and close before doing any work, so we never hold two
        # full (H x W x n_markers) copies simultaneously
        x = da.x.values; y = da.y.values
        data = da.values
        mask = mask_da.values
        da.close(); mask_da.close()
        del da, mask_da; gc.collect()

        # accumulate the dropped markers one at a time: fancy-indexing them all at
        # once would materialize the (y, x, n_dropped) copy this function exists to
        # avoid. log-normalized values are exponentiated before summing so the
        # pseudomarker ends up on the same scale as the markers we kept.
        acc = np.zeros(data.shape[:2], dtype=np.float64)
        for j in drop_ix:
            acc += np.expm1(data[..., j].astype(np.float64))
        other = np.log1p(acc)
        del acc

        # empty pixels are zero in the input, and log1p(sum(expm1(0))) is zero, so
        # they stay empty without any special handling
        s = xr.DataArray(
            np.concatenate([data[..., keep_ix], other[..., None].astype(np.float32)], axis=-1),
            dims=['y', 'x', 'marker'],
            coords={'x': x, 'y': y, 'marker': keep + [pseudomarker]})
        s.name = sid
        util.write_xarray(s, f'{outdir}/{sid}.nc')
        del data, s; gc.collect()

        # a sample with an empty mask contributes nothing rather than a nan that
        # would poison the pooled moments
        vals = other[mask]
        if len(vals) == 0:
            logger.warning(f'{sid} has no non-empty pixels; excluding it from the '
                           f'{pseudomarker} moments.')
        else:
            sample_sids.append(sid)
            sample_means.append(vals.mean(dtype=np.float64))
            sample_stds.append(vals.std(dtype=np.float64))
            sample_npixels.append(len(vals))
        del other, mask, vals; gc.collect()

    if len(sample_sids) == 0:
        raise ValueError(f'No non-empty pixels in any sample, so {pseudomarker} has no moments. '
                         f'Check that {masksdir} holds the masks for {normeddir}.')

    # pool the pseudomarker's moments across samples the same way get_sumstats does
    pseudo_mean, pseudo_std = util.pool_moments(
        pd.DataFrame([sample_means], index=[pseudomarker], columns=sample_sids),
        pd.DataFrame([sample_stds],  index=[pseudomarker], columns=sample_sids),
        sample_npixels)
    pseudo_mean, pseudo_std = pseudo_mean.iloc[0], pseudo_std.iloc[0]
    if pseudo_std == 0:
        # a constant channel stays exactly 0 after standardization rather than
        # dividing by zero downstream
        logger.warning(f'{pseudomarker} has zero variance; setting its std to 1.')
        pseudo_std = 1.
    logger.info(f'{pseudomarker}: mean {pseudo_mean:.3f}, std {pseudo_std:.3f}')

    # the pseudomarker's moments aren't known until every sample has been read, so
    # the files are written above without them and stamped here; this rewrites
    # metadata only, rather than re-collapsing every sample a second time
    keep_ix_ref = [ref_markers.index(m) for m in keep]
    means = np.append(np.asarray(ref_means)[keep_ix_ref], pseudo_mean).astype(np.float32)
    stds  = np.append(np.asarray(ref_stds)[keep_ix_ref],  pseudo_std).astype(np.float32)
    for sid in sids:
        with netCDF4.Dataset(f'{outdir}/{sid}.nc', 'a') as ds:
            v = ds.variables[sid]
            v.setncattr('means', means)
            v.setncattr('stds', stds)

def pca_pixels(outdir, repname, nmetamarkers=10, npixels_to_plot=50000,
               total_n_metapixels=2_000_000, sid_to_covs=None):
    """
    Reduce normalized pixels to a small set of PCA "meta-markers".

    Fits PCA on spatially pooled metapixels across all samples, then projects
    every pixel onto the top `nmetamarkers` components. This compresses the gene
    panel into a compact representation for model training.

    Parameters
    ----------
    outdir
        Directory written by `prepare_merfish`/`prepare_xenium5k`.
    repname
        Name of the representation; PC loadings are saved under
        ``outdir/repname``.
    nmetamarkers
        Number of principal components (meta-markers) to keep.
    total_n_metapixels
        Target number of pooled metapixels used to fit the PCA.
    sid_to_covs
        Optional per-sample covariates, used only for the diagnostic plots.

    Returns
    -------
    tuple
        ``(allpixels_pca, pc_loadings)``: the projected pixels (with a 'sid'
        column) and the gene-by-component loading matrix.
    """
    # prepare directory structure
    masksdir = f'{outdir}/masks'
    normeddir = f'{outdir}/normalized'
    processeddir = f'{outdir}/{repname}'
    os.makedirs(processeddir, exist_ok=True)

    # prepare
    sids = [os.path.splitext(f)[0]
        for f in os.listdir(normeddir) if f.endswith('.nc') and not f.startswith('.')]

    # create metapixels for more accurate PCA
    metapixels, npixels = dimreduce.metapixels_allsamples(normeddir, masksdir, sids,
                                                          total_n_metapixels=total_n_metapixels)

    # PCA the metapixels
    loadings, allmp = dimreduce.pca_metapixels(metapixels.values(), nmetamarkers)
    loadings.to_feather(f'{processeddir}/_pcloadings.feather')
    del metapixels, allmp; gc.collect()

    # apply the PC loadings to plain pixels
    pca = dimreduce.pca_pixels(normeddir, masksdir, loadings, sids)

    # add covariates
    cov_names = add_covs(pca, sid_to_covs)

    if settings.show_plots():
        visualize_pixels(pca, npixels_to_plot, 'metamarkers', cov_names)
    return pca, loadings

def harmonize(allpixels_pca, sid_to_covs=None, npixels_to_plot=50000):
    """
    Batch-correct meta-marker pixels across samples with Harmony.

    Parameters
    ----------
    allpixels_pca
        Projected pixels from `pca_pixels`, with a 'sid' column.
    sid_to_covs
        Optional per-sample covariates to integrate over in addition to sample
        ID.

    Returns
    -------
    DataFrame
        Copy of `allpixels_pca` with the PC columns replaced by their
        Harmony-corrected values.
    """
    import harmonypy as hm

    harmony_cov_names = add_covs(allpixels_pca, sid_to_covs)
    pcs = [c for c in allpixels_pca.columns if c.startswith('PC')]

    logger.info('Running Harmony...')
    harmony_out = hm.run_harmony(allpixels_pca[pcs].values,
                                 allpixels_pca,
                                 harmony_cov_names,
                                 verbose=settings.verbosity >= Verbosity.default)

    harmpixels = allpixels_pca.copy()
    harmpixels[pcs] = harmony_out.Z_corr

    if settings.show_plots():
        visualize_pixels(harmpixels, npixels_to_plot, 'harm. metamarkers', harmony_cov_names)

    return harmpixels

def write_harmonized(outdir, repname, harmpixels):
    """
    Write harmonized meta-marker pixels to per-sample ``.nc`` files.

    Rasterizes the corrected pixels from `harmonize` back into ``(y, x, marker)``
    matrices under ``outdir/repname``, ready to be loaded with `read_samples`.
    """
    masksdir = f'{outdir}/masks'
    processeddir = f'{outdir}/{repname}'
    pcs = [c for c in harmpixels.columns if c.startswith('PC')]
    hpcs = ['h'+c for c in pcs]
    for sid in settings.progress(harmpixels.sid.unique(), name='write harmonized samples'):
        mask = xr.open_dataarray(f'{masksdir}/{sid}.nc')
        pl = harmpixels[harmpixels.sid == sid]
        s_ = np.zeros((*mask.shape, len(hpcs)))
        s_[mask.data] = pl[pcs].values
        s = xr.DataArray(s_,
             dims=['y', 'x', 'marker'],
             coords={'x': mask.x, 'y': mask.y, 'marker': hpcs})
        s.name = sid
        s.to_netcdf(f'{processeddir}/{sid}.nc', encoding={s.name: util.compression}, engine="netcdf4")
        mask.close(); del mask
        gc.collect()

def sanity_checks(outdir, repname, npcs=1, nskip=3):
    """
    Plot diagnostics of the harmonized meta-marker representation.

    Shows all meta-markers of one sample, a histogram of each meta-marker
    pooled over all samples, and spatial maps of the first ``npcs`` meta-markers
    across a subset of samples.

    Parameters
    ----------
    repname
        Representation subdirectory under ``outdir`` written by
        `write_harmonized`.
    npcs
        Number of leading meta-markers to map spatially across samples.
    nskip
        Plot every ``nskip``-th sample in the spatial maps.
    """
    processeddir = f'{outdir}/{repname}'
    sids = [os.path.splitext(f)[0]
        for f in os.listdir(processeddir) if f.endswith('.nc')]

    logger.info('all PCs of one sample')
    da = xr.open_dataarray(f'{processeddir}/{sids[0]}.nc')
    s = da.astype(np.float32)
    da.close(); del da
    s.plot(col='marker', col_wrap=5, vmin=-10, vmax=10, cmap='seismic')
    settings.show(f'all_pcs_{sids[0]}')
    del s

    logger.info('histogram of each pc')
    da0 = xr.open_dataarray(f'{processeddir}/{sids[0]}.nc')
    nmms = len(da0.marker)
    da0.close(); del da0
    chunks = []
    for sid in sids:
        da = xr.open_dataarray(f'{processeddir}/{sid}.nc')
        chunks.append(da.astype(np.float32).data.reshape((-1, nmms)))
        da.close(); del da
    harmpixels = np.concatenate(chunks)
    del chunks
    harmpixels = harmpixels[(harmpixels != 0).sum(axis=1) > 0]
    plt.figure(figsize=(3*4, 2*int(np.ceil(nmms/4))))
    for i in settings.progress(range(nmms), name='histogram of each pc'):
        plt.subplot(int(np.ceil(nmms/4)), 4, i+1)
        plt.hist(harmpixels[:,i], bins=1000)
    plt.tight_layout()
    settings.show('pc_histograms')
    del harmpixels
    gc.collect()

    for i in range(1, npcs+1):
        logger.info(f'PC{i} of several samples')
        fig, axs = plt.subplots(len(sids[::nskip])//5 + 1, 5, figsize=(16, 4*(len(sids[::nskip])//5 + 1)))
        flat_axs = axs.flatten()
        for ax in flat_axs:
            ax.set_visible(False)
        for sid, ax in zip(sids[::nskip], flat_axs):
            ax.set_visible(True)
            da = xr.open_dataarray(f'{processeddir}/{sid}.nc')
            s = da.astype(np.float32)
            da.close(); del da
            vmax = np.percentile(np.abs(s.sel(marker=f'hPC{i}').data), 99)
            s.sel(marker=f'hPC{i}').plot(ax=ax, cmap='seismic', vmin=-vmax, vmax=vmax, add_colorbar=False)
            ax.set_title(sid)
            del s; gc.collect()
        plt.tight_layout()
        settings.show(f'pc{i}_by_sample')