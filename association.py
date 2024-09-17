import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import multianndata as md
from . import model as tm

def anndata(patchmeta, Z, samplemeta, var_names=None, use_rep='X', n_comps=10, sampleid='donor'):
	d = ad.AnnData(Z)
	if var_names is not None:
		d.var_names = var_names
	d.obs = patchmeta

	if use_rep == 'X_pca':
		sc.tl.pca(d, n_comps=min(n_comps, Z.shape[1]-1))

	print('running UMAP')
	sc.pp.neighbors(d, use_rep=use_rep)
	sc.tl.umap(d)

	samplemeta.index = samplemeta.index.astype(str)
	d.obs.sid = d.obs.sid.astype(str)
	d = md.MultiAnnData(d)
	d.samplem = samplemeta
	d.sampleid = sampleid
	print(f'built MultiAnnData object with {sampleid} as the unit of analysis')

	return d

def latentrep(P, model, samplemeta, **kwargs):
	P.augmentation_off()
	rlosses, Z = tm.evaluate(model, P, detailed=True)
	print(f'mean loss = {rlosses.mean()}')

	return anndata(P.meta, Z, samplemeta,
		var_names=[f'L{i}' for i in range(1, Z.shape[1]+1)],
		**kwargs)