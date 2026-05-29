# vima
Variational inference-based microniche analysis is a method for conducting case-control analysis on multi-sample spatial molecular datasets. `vima` can be applied to any spatially resolved molecular technology, is well powered even at the modest sample sizes typical of research cohorts, and avoids traditional, parameter-intensive preprocessing steps such as cell segmentation or clustering of cells into discrete cell types. It works by treating each spatial sample as an image and using a variational autoencoder to extract numerical "fingerprints" from small tissue patches that capture their biological content. It uses these fingerprints to define a large number of "microniches" – small, potentially overlapping groups of tissue patches with highly similar biology that span multiple samples. It then uses rigorous permutation testing to identify microniches whose abundance correlates significantly with case-control status after accounting for multiple testing.

## installation
`vima` can be installed via `pip` as follows:
   ```
   pip install vima-spatial
   ```
Note that `vima` requires `pytorch` and `harmonypy`. These should install automatically through `pip`, but if you have trouble, try installing them first, verifying that they work, and then installing `vima`.

## demo
To get started with an example analysis on a toy spatial transcriptomics dataset, take a look at our brief demo. You can see a [completed read-only version](https://github.com/yakirr/vima/blob/main/demo/demo_ST_minimal.ipynb) or run an [interactive version](https://colab.research.google.com/github/yakirr/tpae/blob/main/demo/demo_ST_minimal.ipynb) yourself on a Google Colab GPU (though this requires Colab Pro due to insufficient memory provided on the free tier.)

To see how to apply `vima` to a stain-based modality like CODEX, immunohistochemistry, or immunofluorescence, look at our [immunofluorescence demo](https://github.com/yakirr/vima/blob/main/demo/demo_IF.ipynb).

## citation
If you use `vima`, please cite:

[Y. Reshef, et al. Powerful and accurate case-control analysis of spatial molecular data. bioRxiv. https://doi.org/10.1101/2025.02.07.637149v1](https://www.biorxiv.org/content/10.1101/2025.02.07.637149v2).
