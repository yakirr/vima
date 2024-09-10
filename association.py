import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import multianndata as md
from . import model as tm

def latentrep(P, model, samplemeta, sampleid='donor'):
	P.augmentation_off()
	rlosses, Z = tm.evaluate(model, P, detailed=True)
	print(f'mean loss = {rlosses.mean()}')

	d = ad.AnnData(Z)
	d.var_names = [f'L{i}' for i in range(1, Z.shape[1]+1)]
	d.obs = P.meta

	print('running UMAP')
	sc.pp.neighbors(d, use_rep='X')
	sc.tl.umap(d)

	samplemeta.index = samplemeta.index.astype(str)
	d.obs.sid = d.obs.sid.astype(str)
	d = md.MultiAnnData(d)
	d.samplem = samplemeta
	d.sampleid = sampleid

	print(f'built MultiAnnData object with {sampleid} as the unit of analysis')

	return d