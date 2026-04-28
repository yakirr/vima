import os, glob, gc
import numpy as np
import cv2 as cv2
import xarray as xr
from . import util
from skimage.filters import threshold_otsu
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

def foreground_mask_codex(s, real_markers, blur_width=5):
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
    totals = (s.sel(marker=real_markers).sum(dim='marker') / (s.sel(marker=neg_ctrls).sum(dim='marker') + len(neg_ctrls) + neg_ctrl_pseudocount))
    totals = cv2.GaussianBlur(totals.data, (blur_width, blur_width),0)
    valid_pixels = totals[(totals > not_imaged_thresh) & (totals < artifact_thresh)]
    t = thresholding_method(valid_pixels)
    
    return xr.DataArray(((totals > t) & (totals < artifact_thresh)).astype('bool'),
                coords={'x': s.x, 'y': s.y},
                dims=['y','x'], name=s.name)

def prepare(load, filepaths, orig_pixel_size, markers, get_foreground, norm_by_background, outdir, pixel_size=10, plot=True):
    pixelsdir = f'{outdir}/counts'
    masksdir = f'{outdir}/masks'
    normeddir = f'{outdir}/normalized'
    os.makedirs(pixelsdir, exist_ok=True)
    os.makedirs(masksdir, exist_ok=True)
    os.makedirs(normeddir, exist_ok=True)
    
    if len(filepaths) == 0:
        print('No files found. Check your filepaths and try again.')
        return

    print('Downsampling...')
    downsample_factor = int(pixel_size//orig_pixel_size)
    for filepath in pb(filepaths):
        sid, sample = load(filepath) # assumes sample is a numpy array of shape (y,x,markers)
        sample = util.hiresarray_to_downsampledxarray(sample,
                                                        sid,
                                                        downsample_factor, orig_pixel_size, markers)
        mask = get_foreground(sample)
        util.write_xarray(sample, f'{pixelsdir}/{sid}.nc')
        util.write_xarray(mask, f'{masksdir}/{sid}.nc')

    print('Computing normalization factor and dataset-wide mean and variance per marker...')
    sids = [os.path.splitext(f)[0]
        for f in os.listdir(pixelsdir) if f.endswith('.nc') and not f.startswith('.')]
    pixels = np.concatenate([
        util.xr_to_pixellist(
            xr.open_dataarray(f'{pixelsdir}/{sid}.nc').astype(np.float32),
            xr.open_dataarray(f'{masksdir}/{sid}.nc')
            )
        for sid in pb(sids)])
    gc.collect()
    _, pixels = norm_by_background(pixels)
    ntranscripts = pixels.sum(axis=1, dtype=np.float64)
    med_ntranscripts = np.median(ntranscripts)
    pixels = np.log1p(med_ntranscripts * pixels / (ntranscripts[:,None] + 1e-6)) # adding to denominator in case pixel is all 0s
    means = pixels.mean(axis=0, dtype=np.float64)
    stds = pixels.std(axis=0, dtype=np.float64)
    del pixels; gc.collect()
    
    print('Normalizing and writing')
    for sid in pb(sids):
        s = xr.open_dataarray(f'{pixelsdir}/{sid}.nc').astype(np.float32)
        mask = xr.open_dataarray(f'{masksdir}/{sid}.nc')
        s = s.where(mask, other=0)
        pl = util.xr_to_pixellist(s, mask)

        goodmarkers, pl = norm_by_background(pl)
        pl = np.log1p(med_ntranscripts * pl / (pl.sum(axis=1)[:,None] + 1e-6)) # adding to denominator in case pixel is all 0s
        pl -= means
        pl /= stds
        s = s.sel(marker=goodmarkers)
        util.set_pixels(s, mask, pl)
        util.write_xarray(s, f'{normeddir}/{sid}.nc')