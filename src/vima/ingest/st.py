import numpy as np
import scanpy as sc
import pandas as pd
import gc, os
from matplotlib import pyplot as plt
from . import util
from .._settings import logger, settings

def med_ntranscripts(load, filepaths, x_col, y_col, pixel_size=10):
    """
    Compute the normalization target: median transcripts per dense pixel.

    Bins each sample's transcripts into ``pixel_size`` square pixels and takes
    the median transcript count over pixels with at least 10 transcripts; returns
    the mean of these per-sample medians, used as the total-count normalization
    target.
    """
    medians = []
    for i, filepath in enumerate(settings.progress(filepaths, name='median #transcripts')):
        sid, df = load(filepath)

        df["px"] = (df[x_col] // pixel_size).astype(np.int32)
        df["py"] = (df[y_col] // pixel_size).astype(np.int32)
        pg = (
            df
            .groupby(["px", "py"])
            .size()
            .reset_index(name="txcount")
        )
        med = pg[pg.txcount > 10].txcount.median()
        logger.info(f"\tSample {i+1}/{len(filepaths)}: {sid} ({len(df)/1e6:.2f}M tx) "
                    f"→ median {med} transcripts per pixel with >=10 transcripts")
        medians.append(med)
        del df, pg; gc.collect()
    return np.mean(np.array(medians))

def get_sumstats(load, filepaths, target_sum, x_col, y_col, gene_col, n_top_genes_per_sample=200, genes_to_add=[],
                 pixel_size=10, min_mean=0.01, min_ntranscripts_per_pixel=10, min_ngenes_per_pixel=1,
                 min_npixels=20, min_totalcounts=500):
    """
    Select highly variable genes and compute dataset-wide per-gene moments.

    For each sample, rasterizes transcripts to pixels, QC-filters pixels,
    log-normalizes to ``target_sum``, and selects the top highly variable genes;
    the union of HVGs across samples (plus any ``genes_to_add``) is returned.
    Also returns the pixel-count-weighted grand mean and grand std per gene
    across the whole dataset (pooling within- and between-sample variance).

    Parameters
    ----------
    load
        Callable mapping a file path to ``(sample_id, transcript_dataframe)``.
    target_sum
        Per-pixel total-count normalization target (from `med_ntranscripts`).
    n_top_genes_per_sample
        Highly variable genes to select per sample; if None, all genes are kept.
    genes_to_add
        Genes to force-include regardless of variability.
    min_mean
        Minimum log-normalized mean for a selected HVG to be retained.

    Returns
    -------
    tuple
        ``(hvgs, grand_mean, grand_std)``: the list of genes to keep and
        per-gene Series of the dataset-wide mean and standard deviation.
    """
    union_hvgs = set()
    union_allgenes = set()
    means = []
    stds = []
    npixels = []

    for i, filepath in enumerate(settings.progress(filepaths, name='HVGs & moments')):
        sid, transcripts = load(filepath)
        ntranscripts = len(transcripts)

        # create pixellist
        pl = util.transcriptlist_to_pixellist(transcripts, x_col, y_col, gene_col, pixel_size=pixel_size)
        del transcripts; gc.collect()
        
        # create scanpy object and filter empty/near-empty pixels
        genes = pl.columns[2:]
        obs = pl[['pixel_x', 'pixel_y']].copy()
        obs.index = obs.index.astype(str)
        pl = sc.AnnData(pl[genes].values.astype(np.float32), var=pd.DataFrame(index=genes), obs=obs)
        sc.pp.filter_cells(pl, min_counts=min_ntranscripts_per_pixel, inplace=True)
        sc.pp.filter_cells(pl, min_genes=min_ngenes_per_pixel, inplace=True)
        
        # compute moments
        X = pl.X / pl.X.sum(axis=1, keepdims=True) * target_sum
        X = np.log1p(X)
        means.append(pd.Series(np.array(X.mean(axis=0, dtype=np.float64)).squeeze(), index=genes))
        stds.append(pd.Series(np.array(X.std(axis=0, dtype=np.float64)).squeeze(), index=genes))
        npixels.append(pl.n_obs)
        union_allgenes.update(genes)

        # QC genes and compute HVGs for this sample
        sc.pp.filter_genes(pl, min_cells=min_npixels, inplace=True)
        sc.pp.filter_genes(pl, min_counts=min_totalcounts, inplace=True)
        if n_top_genes_per_sample is not None:
            if pl.n_vars < n_top_genes_per_sample:
                logger.warning(f'only {pl.n_vars} genes passed QC, which is less than n_top_genes_per_sample = '
                               f'{n_top_genes_per_sample}; skipping HVG selection for this sample and using all '
                               f'{pl.n_vars} genes instead.')
                union_hvgs.update(pl.var_names.tolist())
            else:
                sc.pp.highly_variable_genes(
                    pl,
                    n_top_genes=n_top_genes_per_sample,
                    flavor='seurat_v3',
                    subset=False
                )
                hvgs = pl.var_names[pl.var.highly_variable & (pl.var.means >= min_mean)].tolist()
                hvgs = hvgs + list(set(genes_to_add) & set(pl.var_names))
                
                if settings.show_plots():
                    sc.pl.highly_variable_genes(pl, log=True, show=False)
                    settings.show(f'hvgs_{sid}')

                if settings.show_plots('verbose'):
                    top8 = (
                        pl.var.loc[hvgs]
                        .sort_values('variances_norm', ascending=False)
                        .head(8)
                        .index
                        .tolist()
                    )
                    x = pl.obs['pixel_x'].values
                    y = pl.obs['pixel_y'].values
                    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                    fig.suptitle(sid)
                    for ax, gene in zip(axes.flatten(), top8):
                        expr = pl[:, gene].X
                        if not isinstance(expr, np.ndarray):
                            expr = expr.toarray().ravel()
                        else:
                            expr = expr.ravel()
                        top = np.percentile(expr, 90)
                        if top == 0:
                            top = expr.max() / 2
                        ax.scatter(x[expr==0], y[expr==0], color='gray', s=0.2)
                        ax.scatter(x[expr>0], y[expr>0], c=expr[expr>0], s=0.2, alpha=0.5, cmap='viridis', vmax=top)
                        ax.set_title(gene)
                        ax.set_aspect('equal')
                        ax.axis('off')
                    plt.tight_layout()
                    settings.show(f'top_hvgs_{sid}')
                    plt.close(fig)

                union_hvgs.update(hvgs)
                logger.info(f'\tSample {i+1}/{len(filepaths)}: {sid} ({ntranscripts/1e6:.2f}M tx) → {len(union_hvgs)} HVGs across all samples so far.')
        else:
            union_hvgs.update(genes)
            logger.info(f'\tSample {i+1}/{len(filepaths)}: {sid} ({ntranscripts/1e6:.2f}M tx) → Using all {len(genes)} genes.')

        pl.X = None; del pl; del X; gc.collect()

    means_df = pd.concat([m.reindex(index=union_allgenes, fill_value=0) for m in means], axis=1)
    stds_df  = pd.concat([s.reindex(index=union_allgenes, fill_value=0) for s in stds],  axis=1)
    w = np.array(npixels, dtype=np.float64)
    W = w.sum()

    grand_mean   = np.sum((means_df * w).values, axis=1, dtype=np.float64) / W
    mean_of_vars = np.sum((stds_df ** 2 * w).values, axis=1, dtype=np.float64) / W
    var_of_means = ((means_df.subtract(grand_mean, axis=0).values.astype(np.float64) ** 2) * w).sum(axis=1) / W
    grand_mean = pd.Series(grand_mean, index=means_df.index)
    grand_std  = pd.Series(np.sqrt(mean_of_vars + var_of_means), index=means_df.index)

    return list(union_hvgs), grand_mean, grand_std

def transcriptlist_to_normedpixelmatrix(sid, data, x_col, y_col, gene_col, pixel_size, target_sum, means, stds,
                                  genes, min_ngenes_per_pixel, min_ntranscripts_per_pixel):
    """
    Rasterize one sample's transcripts to a log-normalized ``(y, x, gene)`` matrix.

    Bins transcripts into square pixels, drops pixels failing the QC thresholds,
    log-normalizes surviving pixels to ``target_sum``, restricts to ``genes``
    (filling absent genes with zeros), and returns the pixel matrix alongside its
    tissue mask. The dataset-wide ``means`` and ``stds`` are stored as attributes
    on the matrix for later standardization (they are not applied here).

    Parameters
    ----------
    sid
        Sample ID; used to name the output arrays.
    data
        Transcript table for this sample.
    target_sum
        Per-pixel total-count normalization target.
    means, stds
        Dataset-wide per-gene mean and std, stored on the matrix's attrs.
    genes
        Genes to retain, in output order.

    Returns
    -------
    tuple
        ``(mask, matrix)``: the boolean tissue mask and the normalized
        ``(y, x, gene)`` DataArray.
    """
    logger.info(f'\tNumber of transcripts: {len(data)/1e6:.2f}M')

    # process data
    pl = util.transcriptlist_to_pixellist(
        data,
        x_col,
        y_col,
        gene_col,
        pixel_size=pixel_size
    )
    markers = pl.columns[2:]
    all_x = np.sort(pl['pixel_x'].unique())
    all_y = np.sort(pl['pixel_y'].unique())
    logger.info(f'\tMaking pixel list... {len(pl)} pixels.')

    if settings.show_plots():
        plt.figure(figsize=(5,5))
        plt.scatter(pl.pixel_x, pl.pixel_y, c='gray', s=0.1, alpha=0.2)
    pl = pl[(pl[markers] != 0).sum(axis=1) >= min_ngenes_per_pixel]
    pl = pl[(pl[markers].sum(axis=1) >= min_ntranscripts_per_pixel) ]#& (pl[list(set(markers) & set(genes))].sum(axis=1) > 0)]
    if settings.show_plots():
        plt.scatter(pl.pixel_x, pl.pixel_y, c=pl[markers].sum(axis=1), s=0.1, alpha=0.8, vmin=0, vmax=100)
        plt.gca().set_aspect('equal'); plt.title('transcript density (gray = failed qc)'); plt.axis('off')
        settings.show(f'transcript_density_{sid}')
    logger.info(f'\t{len(pl)} pixels after QC.')

    logger.info('\tLog-normalizing...')
    pl[markers] = pl[markers].div(pl[markers].sum(axis=1), axis=0).fillna(0) * target_sum
    pl[markers] = np.log1p(pl[markers])

    logger.info(f'\trestricting to {len(genes)} genes')
    pl = pl.reindex(columns=['pixel_x', 'pixel_y'] + genes, fill_value=0) # zeros for genes not present in this sample
    mask_pl = pl[['pixel_x', 'pixel_y']].copy()
    mask_pl['nonempty'] = 1
    
    s = util.pixellist_to_pixelmatrix(pl, genes).reindex({'x': all_x, 'y': all_y}, fill_value=0)
    s.attrs['means'] = means.reindex(s.marker.values).values.astype(np.float32)
    s.attrs['stds'] = stds.reindex(s.marker.values).values.astype(np.float32)
    mask = util.pixellist_to_pixelmatrix(mask_pl, ['nonempty']).squeeze().astype(bool).reindex({'x': all_x, 'y': all_y}, fill_value=False)
    s.name = sid; mask.name = sid
    gc.collect()
    logger.info(f'\tMaking pixel matrix... done. shape: {s.shape}')
    
    return mask, s.astype(np.float32)

def rasterize_and_normalize_generic(load, filepaths, x_col, y_col, gene_col, n_top_genes_per_sample, pixel_size, outdir,
                                    min_ntranscripts_per_pixel, min_ngenes_per_pixel,
                                    genes_to_add=[]):
    """
    Shared driver that rasterizes and normalizes every sample to disk.

    Backs `prepare_merfish` and `prepare_xenium5k`: computes the normalization
    factor and dataset-wide gene set/moments once, then rasterizes and
    normalizes each sample, writing its normalized matrix and mask under
    ``outdir/normalized`` and ``outdir/masks``.

    Parameters
    ----------
    load
        Callable mapping a file path to ``(sample_id, transcript_dataframe)``.
    n_top_genes_per_sample
        Highly variable genes per sample; None keeps the full gene panel.
    genes_to_add
        Genes to force-include regardless of variability.
    """
    if len(filepaths) == 0:
        logger.warning('No files found. Check your filepaths and try again.')
        return

    logger.info('Computing normalization factor...')
    normfactor = med_ntranscripts(load, filepaths, x_col, y_col, pixel_size=pixel_size)
    logger.info('Finding HVGs and dataset-wide mean and variance per gene...')
    hvgs, means, stds = get_sumstats(load, filepaths, normfactor, x_col, y_col, gene_col,
                                     n_top_genes_per_sample=n_top_genes_per_sample,
                                     genes_to_add=genes_to_add, pixel_size=pixel_size,
                                     min_ntranscripts_per_pixel=min_ntranscripts_per_pixel,
                                     min_ngenes_per_pixel=min_ngenes_per_pixel)
    logger.info(f'Final number of genes used = {len(hvgs)}')

    logger.info('Rasterizing and normalizing...')
    normdir = f'{outdir}/normalized'
    masksdir = f'{outdir}/masks'
    os.makedirs(normdir, exist_ok=True)
    os.makedirs(masksdir, exist_ok=True)
    for i, filepath in enumerate(settings.progress(filepaths, name='rasterize & normalize')):
        sid, data = load(filepath)
        logger.info(f'Processing sample {i+1}/{len(filepaths)}: {sid}')
        mask, pm = transcriptlist_to_normedpixelmatrix(sid, data, x_col, y_col, gene_col, pixel_size,
                                                       normfactor, means=means, stds=stds, genes=hvgs,
                                                       min_ntranscripts_per_pixel=min_ntranscripts_per_pixel,
                                                       min_ngenes_per_pixel=min_ngenes_per_pixel)
        del data; gc.collect()
        util.write_xarray(mask, f'{masksdir}/{pm.name}.nc')
        util.write_xarray(pm, f'{normdir}/{pm.name}.nc')

def prepare_xenium5k(load, filepaths, x_col, y_col, gene_col, n_top_genes_per_sample, outdir,
                     pixel_size=10, genes_to_add=[],
                     min_ntranscripts_per_pixel=11, min_ngenes_per_pixel=1):
    """
    Rasterize and normalize large-panel (e.g. Xenium 5K) transcript data.

    Like `prepare_merfish`, but selects highly variable genes per sample rather
    than keeping the full panel, which is important for large gene panels. See
    `prepare_merfish` for the shared parameters.

    Parameters
    ----------
    n_top_genes_per_sample
        Number of highly variable genes selected per sample; their union across
        samples is used.
    genes_to_add
        Genes to force-include regardless of variability.
    """
    rasterize_and_normalize_generic(load, filepaths, x_col, y_col, gene_col,
                                  n_top_genes_per_sample,
                                  pixel_size=pixel_size,
                                  outdir=outdir,
                                  genes_to_add=genes_to_add,
                                  min_ntranscripts_per_pixel=min_ntranscripts_per_pixel,
                                  min_ngenes_per_pixel=min_ngenes_per_pixel)

def prepare_merfish(load, filepaths, x_col, y_col, gene_col, outdir,
                    pixel_size=10,
                    min_ntranscripts_per_pixel=11, min_ngenes_per_pixel=1):
    """
    Rasterize and normalize MERFISH-scale transcript data into pixel matrices.

    For each sample, bins transcripts into square pixels, log-normalizes, and
    writes a ``(y, x, gene)`` matrix plus a tissue mask under ``outdir``. The
    full gene panel is kept (no highly-variable-gene selection); use
    `prepare_xenium5k` for large panels. Blank probes and low-quality
    genes/transcripts should be removed before calling.

    Parameters
    ----------
    load
        Callable mapping a file path to ``(sample_id, transcript_dataframe)``.
    filepaths
        Paths to the per-sample transcript files.
    x_col, y_col, gene_col
        Columns in each transcript table giving x/y coordinates and gene name.
    outdir
        Output directory; normalized matrices and masks are written to
        subdirectories.
    pixel_size
        Pixel side length in microns.
    min_ntranscripts_per_pixel, min_ngenes_per_pixel
        Minimum transcripts and distinct genes for a pixel to pass QC.
    """
    rasterize_and_normalize_generic(load, filepaths, x_col, y_col, gene_col,
                                  None,
                                  pixel_size=pixel_size,
                                  outdir=outdir,
                                  min_ntranscripts_per_pixel=min_ntranscripts_per_pixel,
                                  min_ngenes_per_pixel=min_ngenes_per_pixel)