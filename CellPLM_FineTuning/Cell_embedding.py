# -*- coding: utf-8 -*-

import warnings, sys, os 
warnings.filterwarnings("ignore")

import hdf5plugin
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
import scanpy as sc
import matplotlib.pyplot as plt
 # For faster evaluation, we recommend the installation of rapids_singlecell.

"""## Specify important parameters before getting started"""

PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:1'
dataset = 'GSE155468' #or DLPFC


"""## Load Downstream Dataset

The example dataset here is from [GSE155468](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155468).

"""

set_seed(42)
if dataset == 'GSE155468':
    data = ad.read_h5ad('gse155468.h5ad') #add the path to the dataset
    data.obs_names_make_unique()
    data.obs['celltype'] = data.obs['celltype']
elif dataset == 'DLPFC':
    data = ad.read_h5ad('sample_1.h5ad') #specify the sample number
    data.obs_names_make_unique()
    data.obs['celltype'] = data.obs['layer']
else:
    raise ValueError(f"Invalid dataset: {dataset}")



"""## Set up the pipeline"""

pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                 pretrain_directory='ckpt')
pipeline.model

"""## Evaluation and Inference

Once the pipeline is initialized, performing inference (cell embedding query) or evaluation on new datasets (with clustering metrics) can be easily accomplished using the built-in predict and score functions.
"""

embedding = pipeline.predict(data) # Specify a gpu or cpu for model inference

data.obsm['emb'] = embedding.cpu().numpy()
sc.pp.neighbors(data, use_rep='emb') # remove method='rapids' if rapids is not installed
sc.tl.umap(data) # remove method='rapids' if rapids is not installed
plt.rcParams['figure.figsize'] = (6, 6)
sc.pl.umap(data, color='celltype', palette='Paired')

# Drop any rows with NaN in either 'celltype' or 'leiden'
if 'leiden' not in data.obs:
    import scanpy as sc
    sc.tl.leiden(data)  # generate leiden clusters if not present

# Drop NaNs if any
data = data[~(data.obs['celltype'].isna() | data.obs['leiden'].isna())].copy()


result = pipeline.score(data, # An AnnData object
               label_fields=['celltype'],
               evaluation_config = {
                   'method': 'scanpy', # change to 'scanpy' if 'rapids_singlecell' is not installed; the final scores may vary due to the implementation
                   'batch_size': 50000, # Specify batch size to limit gpu memory usage
               }) # Specify a gpu or cpu for model inference

print(result)