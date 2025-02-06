# vima
Variational inference-based microniche analysis is a method for conducting case-control analysis on multi-sample spatial molecular datasets. `vima` can be applied to any spatially resolved molecular technology, is well powered even at the modest sample sizes typical of research cohorts, and avoids traditional, parameter-intensive preprocessing steps such as cell segmentation or clustering of cells into discrete cell types. It works by treating each spatial sample as an image and using a variational autoencoder to extract numerical "fingerprints" from small tissue patches that capture their biological content. It uses these fingerprints to define a large number of "microniches'' – small, potentially overlapping groups of tissue patches with highly similar biology that span multiple samples. It then uses rigorous statistics to identify microniches whose abundance correlates with case-control status.

## installation
To use `vima`, please clone this repository and add it to your `PYTHONPATH`. You will first need to install `pytorch'.

## demo
Coming soon!
