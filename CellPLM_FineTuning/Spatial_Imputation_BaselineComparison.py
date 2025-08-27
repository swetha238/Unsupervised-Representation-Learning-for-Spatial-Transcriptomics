# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import hdf5plugin
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.utils.data import stratified_sample_genes_by_sparsity
from CellPLM.pipeline.imputation import ImputationPipeline, ImputationDefaultPipelineConfig, ImputationDefaultModelConfig

"""## Specify important parameters before getting started"""

DATASET = 'Liver' # 'Lung'
PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:4'


set_seed(11)
if DATASET == 'Lung':
    query_dataset = 'HumanLungCancerPatient2_filtered_ensg.h5ad'
    ref_dataset = 'GSE131907_Lung_ensg.h5ad'
    query_data = ad.read_h5ad(f'../data/{query_dataset}')
    ref_data = ad.read_h5ad(f'../data/{ref_dataset}')

elif DATASET == 'Liver':
    query_dataset = 'HumanLiverCancerPatient2_filtered_ensg.h5ad'
    ref_dataset = 'GSE151530_Liver_ensg.h5ad'
    query_data = ad.read_h5ad(f'../data/{query_dataset}')
    ref_data = ad.read_h5ad(f'../data/{ref_dataset}')

target_genes = stratified_sample_genes_by_sparsity(query_data, seed=11) # This is for reproducing the hold-out gene lists in our paper
query_data.obsm['truth'] = query_data[:, target_genes].X.toarray()
query_data[:, target_genes].X = 0
train_data = query_data.concatenate(ref_data, join='outer', batch_key=None, index_unique=None)

train_data.obs['split'] = 'train'
train_data.obs['split'][train_data.obs['batch']==query_data.obs['batch'][-1]] = 'valid'
train_data.obs['split'][train_data.obs['batch']==ref_data.obs['batch'][-1]] = 'valid'

"""## Specify gene to impute
In the last step, we merge the query dataset (SRT) and the reference dataset (scRNA-seq). However, the query dataset does not measures all the genes. For fine-tuning the model, we need to specify which genes are measured in each dataset. Therefore, we create a dictionary for the imputation pipeline.
"""

query_genes = [g for g in query_data.var.index if g not in target_genes]
query_batches = list(query_data.obs['batch'].unique())
ref_batches = list(ref_data.obs['batch'].unique())
batch_gene_list = dict(zip(list(query_batches) + list(ref_batches),
    [query_genes]*len(query_batches) + [ref_data.var.index.tolist()]*len(ref_batches)))

"""## Overwrite parts of the default config
These hyperparameters are recommended for general purpose. We did not tune it for individual datasets. You may update them if needed.
"""

pipeline_config = ImputationDefaultPipelineConfig.copy()
model_config = ImputationDefaultModelConfig.copy()

pipeline_config, model_config

"""## Fine-tuning

Efficient data setup and fine-tuning can be seamlessly conducted using the CellPLM built-in `pipeline` module.

First, initialize a `ImputationPipeline`. This pipeline will automatically load a pretrained model.
"""

pipeline = ImputationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                      overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                      pretrain_directory='../ckpt')
pipeline.model

"""Next, employ the `fit` function to fine-tune the model on your downstream dataset. This dataset should be in the form of an AnnData object, where `.X` is a csr_matrix. See previous section for more details.

Typically, a dataset containing approximately 20,000 cells can be trained in under 10 minutes using a V100 GPU card, with an expected GPU memory consumption of around 8GB.
"""

pipeline.fit(train_data, # An AnnData object
            pipeline_config, # The config dictionary we created previously, optional
            split_field = 'split', #  Specify a column in .obs that contains split information
            train_split = 'train',
            valid_split = 'valid',
            batch_gene_list = batch_gene_list, # Specify genes that are measured in each batch, see previous section for more details
            device = DEVICE,
            )

"""## Inference and evaluation
Once the pipeline has been fitted to the downstream datasets, performing inference or evaluation on new datasets can be easily accomplished using the built-in `predict` and `score` functions.
"""

pipeline.predict(
        query_data, # An AnnData object
        pipeline_config, # The config dictionary we created previously, optional
        device = DEVICE,
    )

pipeline.score(
                query_data, # An AnnData object
                evaluation_config = {'target_genes': target_genes}, # The config dictionary we created previously, optional
                label_fields = ['truth'], # A field in .obsm that stores the ground-truth for evaluation
                device = DEVICE,
)

