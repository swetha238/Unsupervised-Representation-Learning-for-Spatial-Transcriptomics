import warnings
warnings.filterwarnings("ignore")

import hdf5plugin
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.pipeline.cell_type_annotation import CellTypeAnnotationPipeline, CellTypeAnnotationDefaultPipelineConfig, CellTypeAnnotationDefaultModelConfig
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score, f1_score


PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:3'
DATASET = 'hPancreas' # or 'MS'

if DATASET == 'hPancreas':
    data_train = ad.read_h5ad(f'../data/demo_train.h5ad')
    data_test = ad.read_h5ad(f'../data/demo_test.h5ad')
    train_num = data_train.shape[0]
    data = ad.concat([data_train, data_test])
    data.X = csr_matrix(data.X)
    data.obs['celltype'] = data.obs['Celltype']

elif DATASET == 'MS':
    data_train = ad.read_h5ad(f'../data/c_data.h5ad')
    data_test = ad.read_h5ad(f'../data/filtered_ms_adata.h5ad')
    data_train.var = data_train.var.set_index('index_column')
    data_test.var = data_test.var.set_index('index_column')
    train_num = data_train.shape[0]
    data = ad.concat([data_train, data_test])
    data.var_names_make_unique()

"""## Overwrite parts of the default config
These hyperparameters are recommended for general purpose. We did not tune it for individual datasets. You may update them if needed.
"""


pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()
pipeline_config['epochs'] = 600
model_config = CellTypeAnnotationDefaultModelConfig.copy()
model_config['out_dim'] = data.obs['celltype'].nunique()


pipeline = CellTypeAnnotationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                      overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                      pretrain_directory='ckpt')
pipeline.model

"""Next, employ the `fit` function to fine-tune the model on your downstream dataset. This dataset should be in the form of an AnnData object, where `.X` is a csr_matrix, and `.obs` includes information for train-test splitting and cell type labels.

Typically, a dataset containing approximately 20,000 cells can be trained in under 10 minutes using a V100 GPU card, with an expected GPU memory consumption of around 8GB.
"""

pipeline.fit(data, # An AnnData object
            pipeline_config, # The config dictionary we created previously, optional
            split_field = 'split', #  Specify a column in .obs that contains split information
            train_split = 'train',
            valid_split = 'valid',
            label_fields = ['celltype']) # Specify a column in .obs that contains cell type labels




pipeline.predict(
                data, # An AnnData object
                pipeline_config, # The config dictionary we created previously, optional
            )


results = pipeline.score(data, # An AnnData object
                pipeline_config, # The config dictionary we created previously, optional
                split_field = 'split', # Specify a column in .obs to specify train and valid split, optional
                target_split = 'test', # Specify a target split to predict, optional
                label_fields = ['celltype']) 
print(results)              
# Predict on test data
pred_labels = pipeline.predict(
    data_test, 
    pipeline_config
)

# Inverse transform to get actual label names
pred_names = pipeline.label_encoders['celltype'].inverse_transform(pred_labels)
data_test.obs['predicted_celltype'] = pred_names    

# Confusion Matrix
data_test.obs['celltype'] = data_test.obs['Celltype']

true = data_test.obs['celltype']
ConfusionMatrixDisplay.from_predictions(true, pred_names, xticks_rotation=90, cmap='viridis')

# UMAP
import scanpy as sc  # Make sure this is imported
sc.pp.neighbors(data_test)
sc.tl.umap(data_test)
umap_plot = sc.pl.umap(data_test, color=['celltype', 'predicted_celltype'], wspace=0.4)
umap_plot.savefig("umap_plot1.png")

# Classification report
print(classification_report(true, pred_names))
print("Accuracy:", accuracy_score(true, pred_names))
print("F1 score:", f1_score(true, pred_names, average='weighted'))
