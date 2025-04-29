# vima
Variational inference-based microniche analysis is a method for conducting case-control analysis on multi-sample spatial molecular datasets. `vima` can be applied to any spatially resolved molecular technology, is well powered even at the modest sample sizes typical of research cohorts, and avoids traditional, parameter-intensive preprocessing steps such as cell segmentation or clustering of cells into discrete cell types. It works by treating each spatial sample as an image and using a variational autoencoder to extract numerical "fingerprints" from small tissue patches that capture their biological content. It uses these fingerprints to define a large number of "microniches" – small, potentially overlapping groups of tissue patches with highly similar biology that span multiple samples. It then uses rigorous statistics to identify microniches whose abundance correlates with case-control status.

## installation
To use `vima`, you can either install it directly from the [Python Package Index](https://pypi.org/) by running, e.g.,

`pip install vima-spatial`

or if you'd like to manipulate the source code you can clone this repository and add it to your `PYTHONPATH`.

Note that the package requires a working installation of `pytorch`, and it may be beneficial to first install `pytorch`, verify it works properly, and then install `vima`. For data preprocessing the current version of the package also requires a working `R` environment with the [`harmony` package](https://github.com/immunogenomics/harmony) installed. You will need the path to the `Rscript` executable in your `R` environment, which you can obtain by running `which Rscript` with the `R` environment active.

## demo
Take a look at our [demo](https://github.com/yakirr/vima/blob/main/demo/demo_IHC.ipynb) to see how to get started with an example analysis. We plan to put up demos for other data modalities in the future.

## citation
If you use `vima`, please cite:

[Y. Reshef, et al. Powerful and accurate case-control analysis of spatial molecular data with deep learning-defined tissue microniches. bioRxiv. https://doi.org/10.1101/2025.02.07.637149v1](https://www.biorxiv.org/content/10.1101/2025.02.07.637149v1).
