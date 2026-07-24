"""Utilities for fetching example/demo datasets."""

import os
import json
from urllib.request import urlopen, Request

from .._settings import settings, logger

__all__ = ["download_zenodo"]


def download_zenodo(record_id, target_dir, chunk_size=1024 * 1024):
    """Download every file attached to a Zenodo record into ``target_dir``.

    Parameters
    ----------
    record_id
        Zenodo record id -- the number in the record URL, e.g. ``"21535534"``.
    target_dir
        Directory to write the files into; created if it does not exist.
    chunk_size
        Streaming chunk size in bytes.
    """
    os.makedirs(target_dir, exist_ok=True)
    with urlopen(f"https://zenodo.org/api/records/{record_id}") as r:
        files = json.load(r)["files"]
    for i, f in enumerate(files):
        url = f["links"]["self"]
        fname = f["key"]
        size = f.get("size", 0)
        dest = os.path.join(target_dir, fname)
        logger.info(f"[{i + 1}/{len(files)}] downloading {fname}")
        req = Request(url, headers={"User-Agent": "vima"})
        with urlopen(req) as resp, open(dest, "wb") as out, \
                settings.progress(total=size, name=fname,
                                  unit="B", unit_scale=True) as pbar:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                pbar.update(len(chunk))

def download_toy_rawdata(target_dir):
    """Download the toy raw data for the demo.

    Parameters
    ----------
    target_dir
        Directory to write the files into; created if it does not exist.
    """
    download_zenodo("20433752", target_dir)

def download_toy_metadata_and_fingerprints(target_dir):
    """Download the toy metadata and precomputed fingerprints for the demo.

    Parameters
    ----------
    target_dir
        Directory to write the files into; created if it does not exist.
    """
    download_zenodo("21535534", target_dir)