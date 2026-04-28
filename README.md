# vima
Variational inference-based microniche analysis is a method for conducting case-control analysis on multi-sample spatial molecular datasets. `vima` can be applied to any spatially resolved molecular technology, is well powered even at the modest sample sizes typical of research cohorts, and avoids traditional, parameter-intensive preprocessing steps such as cell segmentation or clustering of cells into discrete cell types. It works by treating each spatial sample as an image and using a variational autoencoder to extract numerical "fingerprints" from small tissue patches that capture their biological content. It uses these fingerprints to define a large number of "microniches" – small, potentially overlapping groups of tissue patches with highly similar biology that span multiple samples. It then uses rigorous permutation testing to identify microniches whose abundance correlates significantly with case-control status after accounting for multiple testing.

## installation
`vima` can be installed via `pip` as follows:
   ```
   pip install vima-spatial
   ```

## demo
Take a look at our [demo](https://github.com/yakirr/vima/blob/main/demo/demo_IF.ipynb) to see how to get started with an example analysis. We plan to put up demos for other data modalities in the future.

## citation
If you use `vima`, please cite:

[Y. Reshef, et al. Powerful and accurate case-control analysis of spatial molecular data. bioRxiv. https://doi.org/10.1101/2025.02.07.637149v1](https://www.biorxiv.org/content/10.1101/2025.02.07.637149v2).
