import os, glob, gc
import numpy as np
import pandas as pd
import cv2 as cv2
import xarray as xr
from . import util
from skimage.filters import threshold_otsu
from .._settings import settings, logger

def foreground_mask_codex(s, real_markers, blur_width=5):
    """
    Compute a foreground/tissue mask for a CODEX image.

    Sums the real-marker channels, log-scales to 8-bit intensity, Gaussian
    blurs, and applies Otsu thresholding to separate tissue from background.

    Parameters
    ----------
    real_markers
        Markers (excluding blanks/controls) summed into the total-intensity
        image.

    Returns
    -------
    DataArray
        Boolean ``(y, x)`` mask, True over tissue.
    """
    # compute total intensity
    totals = s.sel(marker=real_markers).sum(dim='marker')
    totals = np.log1p(totals)
    totals -= totals.min()
    totals /= (totals.max()/255)
    totals = totals.astype('uint16')

    # determine foreground vs background
    blurred = cv2.GaussianBlur(totals.data,(blur_width, blur_width),0)
    _, mask = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return xr.DataArray(mask.astype('bool'),
                coords={'x': totals.x, 'y': totals.y},
                dims=['y','x'], name=s.name)

def foreground_mask_if(s, real_markers, neg_ctrls, not_imaged_thresh, artifact_thresh, thresholding_method=threshold_otsu,
        neg_ctrl_pseudocount=0, blur_width=5):
    """
    Compute a foreground/tissue mask for an immunofluorescence image.

    Forms a background-normalized total-intensity image (real markers divided by
    negative-control markers), Gaussian blurs it, and thresholds it. Pixels
    below ``not_imaged_thresh`` (not imaged) or above ``artifact_thresh``
    (artifacts) are excluded when fitting the threshold and from the mask.

    Parameters
    ----------
    real_markers
        Signal markers summed into the numerator.
    neg_ctrls
        Negative-control markers summed into the denominator.
    not_imaged_thresh, artifact_thresh
        Lower/upper intensity bounds; pixels outside are treated as unimaged or
        artifactual.
    thresholding_method
        Callable mapping the valid pixel intensities to a scalar threshold.

    Returns
    -------
    DataArray
        Boolean ``(y, x)`` mask, True over tissue.
    """
    totals = (s.sel(marker=real_markers).sum(dim='marker') / (s.sel(marker=neg_ctrls).sum(dim='marker') + len(neg_ctrls) + neg_ctrl_pseudocount))
    totals = cv2.GaussianBlur(totals.data, (blur_width, blur_width),0)
    valid_pixels = totals[(totals > not_imaged_thresh) & (totals < artifact_thresh)]
    t = thresholding_method(valid_pixels)
    
    return xr.DataArray(((totals > t) & (totals < artifact_thresh)).astype('bool'),
                coords={'x': s.x, 'y': s.y},
                dims=['y','x'], name=s.name)

def prepare(load, filepaths, orig_pixel_size, markers, get_foreground, norm_by_background, outdir, pixel_size=10):
    """
    Rasterize and normalize non-transcriptomics image data into pixel matrices.

    The imaging analogue of `prepare_merfish`: downsamples each sample's
    hi-res ``(y, x, marker)`` array to ``pixel_size``, computes a foreground
    mask, then background-normalizes, total-count-normalizes, and log-scales
    every pixel using a dataset-wide count target and per-marker mean/std.
    Writes downsampled counts, masks, and normalized matrices under ``outdir``.

    Parameters
    ----------
    load
        Callable mapping a file path to ``(sample_id, array)``, where ``array``
        is a ``(y, x, marker)`` numpy array.
    orig_pixel_size
        Pixel side length in microns of the input images.
    markers
        Marker names labeling the array's channel axis.
    get_foreground
        Callable mapping a downsampled sample to its boolean tissue mask (e.g.
        `foreground_mask_codex` / `foreground_mask_if`).
    norm_by_background
        Callable mapping a pixel-by-marker array to ``(kept_markers, normalized_array)``,
        correcting for background/autofluorescence.
    pixel_size
        Target pixel side length in microns.
    """
    pixelsdir = f'{outdir}/counts'
    masksdir = f'{outdir}/masks'
    normeddir = f'{outdir}/normalized'
    os.makedirs(pixelsdir, exist_ok=True)
    os.makedirs(masksdir, exist_ok=True)
    os.makedirs(normeddir, exist_ok=True)
    
    if len(filepaths) == 0:
        logger.warning('No files found. Check your filepaths and try again.')
        return

    logger.info('Downsampling...')
    downsample_factor = int(pixel_size//orig_pixel_size)
    for filepath in settings.progress(filepaths):
        sid, sample = load(filepath) # assumes sample is a numpy array of shape (y,x,markers)
        sample = util.hiresarray_to_downsampledxarray(sample,
                                                        sid,
                                                        downsample_factor, orig_pixel_size, markers)
        mask = get_foreground(sample)
        util.write_xarray(sample, f'{pixelsdir}/{sid}.nc')
        util.write_xarray(mask, f'{masksdir}/{sid}.nc')

    logger.info('Computing normalization factor and dataset-wide mean and variance per marker...')
    sids = [os.path.splitext(f)[0]
        for f in os.listdir(pixelsdir) if f.endswith('.nc') and not f.startswith('.')]
    pixels = np.concatenate([
        util.xr_to_pixellist(
            xr.open_dataarray(f'{pixelsdir}/{sid}.nc').astype(np.float32),
            xr.open_dataarray(f'{masksdir}/{sid}.nc')
            )
        for sid in settings.progress(sids)])
    gc.collect()
    goodmarkers, pixels = norm_by_background(pixels)
    ntranscripts = pixels.sum(axis=1, dtype=np.float64)
    med_ntranscripts = np.median(ntranscripts)
    pixels = np.log1p(med_ntranscripts * pixels / (ntranscripts[:,None] + 1e-6)) # adding to denominator in case pixel is all 0s
    # indexed by marker name so each sample's moments are aligned by name below,
    # rather than positionally against whatever markers that sample kept
    means = pd.Series(pixels.mean(axis=0, dtype=np.float64), index=goodmarkers)
    stds = pd.Series(pixels.std(axis=0, dtype=np.float64), index=goodmarkers)
    del pixels; gc.collect()
    
    logger.info('Normalizing and writing')
    for sid in settings.progress(sids):
        s = xr.open_dataarray(f'{pixelsdir}/{sid}.nc').astype(np.float32)
        mask = xr.open_dataarray(f'{masksdir}/{sid}.nc')
        s = s.where(mask, other=0)
        pl = util.xr_to_pixellist(s, mask)

        goodmarkers, pl = norm_by_background(pl)
        pl = np.log1p(med_ntranscripts * pl / (pl.sum(axis=1)[:,None] + 1e-6)) # adding to denominator in case pixel is all 0s
        s = s.sel(marker=goodmarkers)
        util.set_pixels(s, mask, pl)
        # a marker with no moments standardizes to a no-op rather than dividing by zero
        s.attrs['means'] = means.reindex(s.marker.values, fill_value=0).values.astype(np.float32)
        s.attrs['stds'] = stds.reindex(s.marker.values, fill_value=1).values.astype(np.float32)
        util.write_xarray(s, f'{normeddir}/{sid}.nc')