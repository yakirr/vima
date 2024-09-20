import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import multianndata as md
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

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

def apply(model, P, embedding=None, batch_size=1000):
	if embedding is None:
		embedding = model.embedding

	P.pytorch_mode()
	P.augmentation_off()
	model.eval()
	eval_loader = DataLoader(
		dataset=P,
		batch_size=batch_size,
		shuffle=False)

	Z = []
	with torch.no_grad():
		for batch in pb(eval_loader):
			Z.append(embedding(batch).detach().cpu().numpy())

	return np.concatenate(Z)

def latentrep(model, P, samplemeta, **kwargs):
	apply(model, P)

	return anndata(P.meta, Z, samplemeta,
		var_names=[f'L{i}' for i in range(1, Z.shape[1]+1)],
		**kwargs)