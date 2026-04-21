import numpy as np
import scanpy as sc
import pandas as pd
import gc, os
from matplotlib import pyplot as plt
from . import util

def med_ntranscripts(load, filepaths, x_col, y_col, pixel_size=10):
    medians = []
    for i, filepath in enumerate(filepaths):
        print(f"\tProcessing sample {i+1}/{len(filepaths)}:", end=" ")
        sid, df = load(filepath)
        print(f"{sid} ({len(df)/1e6:.2f}M tx)", end=" ")

        df["px"] = (df[x_col] // pixel_size).astype(np.int32)
        df["py"] = (df[y_col] // pixel_size).astype(np.int32)
        pg = (
            df
            .groupby(["px", "py"])
            .size()
            .reset_index(name="txcount")
        )
        med = pg[pg.txcount > 10].txcount.median()
        print(f"→ median {med} transcripts per pixel with >=10 transcripts")
        medians.append(med)
        del df, pg; gc.collect()
    return np.mean(np.array(medians))

def get_sumstats(load, filepaths, target_sum, x_col, y_col, gene_col, n_top_genes_per_sample=200, genes_to_add=[],
                 pixel_size=10, min_mean=0.01, min_ntranscripts=10, min_npixels=20, min_totalcounts=500, plot=True):
    union_hvgs = set()
    union_allgenes = set()
    means = []
    stds = []

    for i, filepath in enumerate(filepaths):
        print(f'\tProcessing sample {i+1}/{len(filepaths)}:', end=' ')
        sid, transcripts = load(filepath)
        print(f'{sid} has {len(transcripts)/1e6:.2f}M transcripts.', end=' ')
        
        # create pixellist
        pl = util.transcriptlist_to_pixellist(transcripts, x_col, y_col, gene_col, pixel_size=pixel_size)
        del transcripts; gc.collect()
        
        # create scanpy object and filter empty/near-empty pixels
        genes = pl.columns[2:]
        pl = sc.AnnData(pl[genes].values.astype(np.float32), var=pd.DataFrame(index=genes), obs=pl[['pixel_x', 'pixel_y']])
        sc.pp.filter_cells(pl, min_counts=min_ntranscripts, inplace=True)
        
        # compute moments
        X = pl.X / pl.X.sum(axis=1, keepdims=True) * target_sum
        X = np.log1p(X)
        means.append(pd.Series(np.array(X.mean(axis=0, dtype=np.float64)).squeeze(), index=genes))
        stds.append(pd.Series(np.array(X.std(axis=0, dtype=np.float64)).squeeze(), index=genes))
        union_allgenes.update(genes)

        # QC genes and compute HVGs for this sample
        sc.pp.filter_genes(pl, min_cells=min_npixels, inplace=True)
        sc.pp.filter_genes(pl, min_counts=min_totalcounts, inplace=True)
        if n_top_genes_per_sample is not None:
            if pl.n_vars < n_top_genes_per_sample:
                print(f'\033[91m\n\tWARNING: only {pl.n_vars} genes passed QC, which is less than n_top_genes_per_sample = {n_top_genes_per_sample}\033[0m')
                print(f'\tWill skip HVG selection for this sample and use all {pl.n_vars} genes instead.')
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
                
                if plot:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    # Left panel: histogram of mean expression
                    axes[0].hist(np.log10(pl.var.means), bins=50, alpha=0.8, edgecolor='black')
                    axes[0].axvline(np.log10(min_mean), color='red', linestyle='--', label='min_mean threshold')
                    axes[0].set_xlabel('log(Mean counts)')
                    axes[0].set_ylabel('Number of genes')
                    axes[0].set_title(f'{sid} - Mean expression distribution')
                    
                    # Right panel: mean vs variance scatter plot
                    axes[1].scatter(pl.var.means, np.sqrt(pl.var.variances), alpha=0.8, s=4)
                    axes[1].scatter(pl.var.loc[hvgs, 'means'], np.sqrt(pl.var.loc[hvgs, 'variances']), alpha=0.8, s=4)
                    axes[1].set_yscale('log')
                    for gene in hvgs:
                        mean = pl.var.loc[gene, 'means']
                        var = pl.var.loc[gene, 'variances']
                        axes[1].text(mean, np.sqrt(var), gene, fontsize=8, alpha=0.7)
                    axes[1].set_xlabel('Mean counts')
                    axes[1].set_ylabel('Std. Dev.')
                    axes[1].set_title(sid)    
                    plt.tight_layout()
                    plt.show(); plt.close(fig)
                    sc.pl.highly_variable_genes(pl, log=True)
                    plt.show()

                    top20 = (
                        pl.var
                        .sort_values('means', ascending=False)
                        .head(20)[['means', 'highly_variable']]
                    )
                    print(top20)

                    top20_genes = (
                        pl.var
                        .sort_values('means', ascending=False)
                        .head(20)
                        .index
                        .tolist()
                    )
                    x = pl.obs['pixel_x']
                    y = pl.obs['pixel_y']

                    for gene in top20_genes:
                        expr = pl[:, gene].X

                        # handle sparse matrices
                        if not isinstance(expr, np.ndarray):
                            expr = expr.toarray().ravel()
                        else:
                            expr = expr.ravel()

                        plt.figure(figsize=(4,4))
                        plt.scatter(x, y, c=expr, s=2, cmap='viridis', vmax=np.percentile(expr, 90))
                        plt.title(gene)
                        plt.colorbar(label='expression')
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.tight_layout()
                        plt.show()


                union_hvgs.update(hvgs)
                print(f'\033[92m\t{len(union_hvgs)} HVGs across all samples so far.\033[0m')
        else:
            union_hvgs.update(genes)
            print(f'Using all {len(genes)} genes.')

        pl.X = None; del pl; del X; gc.collect()

    means = pd.concat([m.reindex(index=union_allgenes) for m in means], axis=1)
    stds = pd.concat([s.reindex(index=union_allgenes) for s in stds], axis=1)
    return list(union_hvgs), means.mean(axis=1), stds.mean(axis=1)

def transcriptlist_to_normedpixelmatrix(sid, data, x_col, y_col, gene_col, pixel_size, target_sum, means, stds,
                                  genes=None, min_ngenes_per_pixel=5, min_ntranscripts_per_pixel=10, plot=True):
    print(f'\tNumber of transcripts: {len(data)/1e6:.2f}M')
    
    # process data
    print('\tMaking pixel list...', end='')
    pl = util.transcriptlist_to_pixellist(
        data,
        x_col,
        y_col,
        gene_col,
        pixel_size=pixel_size
    )
    markers = pl.columns[2:]
    print(f'{len(pl)} pixels.')

    if plot:
        plt.figure(figsize=(5,5))
        plt.scatter(pl.pixel_x, pl.pixel_y, c='gray', s=0.1, alpha=0.2)
    pl = pl[(pl[markers] != 0).sum(axis=1) >= min_ngenes_per_pixel]
    pl = pl[(pl[markers].sum(axis=1) >= min_ntranscripts_per_pixel) & (pl[list(set(markers) & set(genes))].sum(axis=1) > 0)]
    if plot:
        plt.scatter(pl.pixel_x, pl.pixel_y, c=pl[markers].sum(axis=1), s=0.1, alpha=0.8, vmin=0, vmax=100)
        plt.gca().set_aspect('equal'); plt.title('transcript density (gray = failed qc)'); plt.axis('off'); plt.show()
    print(f'\t{len(pl)} pixels after QC.')

    print('\tLog-normalizing and centering...')
    pl[markers] = pl[markers].div(pl[markers].sum(axis=1), axis=0).fillna(0) * target_sum
    pl[markers] = np.log1p(pl[markers])
    pl = pl.reindex(columns=['pixel_x', 'pixel_y'] + list(means.index), fill_value=0).copy() # add 0s for genes that didn't show up in this sample
    pl[markers] = (pl - means)[markers]
    pl[markers] = (pl / stds)[markers]
    
    if genes is not None:
        print(f'\trestricting to {len(genes)} genes')
        pl = pl.reindex(columns=['pixel_x', 'pixel_y'] + genes, fill_value=0)
    mask_pl = pl[['pixel_x', 'pixel_y']].copy()
    mask_pl['nonempty'] = 1
    
    print('\tMaking pixel matrix...', end='')
    s = util.pixellist_to_pixelmatrix(pl, genes)
    mask = util.pixellist_to_pixelmatrix(mask_pl, ['nonempty']).squeeze().astype(bool)
    s.name = sid; mask.name = sid
    gc.collect()
    print('done. shape:', s.shape)
    
    return mask, s.astype(np.float32)

def rasterize_and_normalize_generic(load, filepaths, x_col, y_col, gene_col, n_top_genes_per_sample, pixel_size, outdir,
                                    genes_to_add=[], plot=True):
    print('Computing normalization factor...')
    normfactor = med_ntranscripts(load, filepaths, x_col, y_col, pixel_size=pixel_size)
    print('Finding HVGs and dataset-wide mean and variance per gene...')
    hvgs, means, stds = get_sumstats(load, filepaths, normfactor, x_col, y_col, gene_col,
                                     n_top_genes_per_sample=n_top_genes_per_sample,
                                     genes_to_add=genes_to_add, pixel_size=pixel_size, plot=plot)
    print('Final number of genes used =', len(hvgs))

    print('Rasterizing and normalizing...')
    normdir = f'{outdir}/normalized'
    masksdir = f'{outdir}/masks'
    os.makedirs(normdir, exist_ok=True)
    os.makedirs(masksdir, exist_ok=True)
    for i, filepath in enumerate(filepaths):
        sid, data = load(filepath)
        print(f'Processing sample {i+1}/{len(filepaths)}: {sid}')
        mask, pm = transcriptlist_to_normedpixelmatrix(sid, data, x_col, y_col, gene_col, pixel_size, normfactor, means, stds, genes=hvgs, plot=plot)
        del data; gc.collect()
        util.write_xarray(mask, f'{masksdir}/{pm.name}.nc')
        util.write_xarray(pm, f'{normdir}/{pm.name}.nc')

def prepare_xenium5k(load, filepaths, x_col, y_col, gene_col, n_top_genes_per_sample, outdir, pixel_size=10, genes_to_add=[], plot=True):
    rasterize_and_normalize_generic(load, filepaths, x_col, y_col, gene_col,
                                  n_top_genes_per_sample,
                                  pixel_size=pixel_size,
                                  outdir=outdir,
                                  genes_to_add=genes_to_add,
                                  plot=plot)

def prepare_merfish(load, filepaths, x_col, y_col, gene_col, outdir, pixel_size=10, plot=True):
    rasterize_and_normalize_generic(load, filepaths, x_col, y_col, gene_col,
                                  None,
                                  pixel_size=pixel_size,
                                  outdir=outdir,
                                  plot=plot)