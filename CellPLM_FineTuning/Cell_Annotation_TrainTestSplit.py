import warnings, sys, os
warnings.filterwarnings("ignore")

import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.pipeline.cell_type_annotation import (
    CellTypeAnnotationPipeline,
    CellTypeAnnotationDefaultPipelineConfig,
    CellTypeAnnotationDefaultModelConfig,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

# CONFIG
PRETRAIN_VERSION = '20230926_85M'
DEVICE = 'cuda:3'
SEED = 42
N_FOLDS = 5  # Number of CV folds
set_seed(SEED)
dataset = 'MERFISH' # or DLPFC

# LOAD ALL FILES
if dataset == 'MERFISH':
    file_paths = [f"MERFISH_{i}.h5ad" for i in range(1, 6)] #add the path to the dataset
elif dataset == 'DLPFC':
    file_paths = [f"sample_{i}.h5ad" for i in range(1, 13)] #add the path to the dataset
else:
    raise ValueError(f"Invalid dataset: {dataset}")

for p in file_paths:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing file: {p}")

def drop_dups(adata):
    if adata.var.index.duplicated().any():
        mask = ~adata.var.index.duplicated()
        return adata[:, mask].copy()
    return adata

adatas = [drop_dups(ad.read_h5ad(p)) for p in file_paths]

# INTERSECT GENES
common = set(adatas[0].var.index)
for A in adatas[1:]:
    common &= set(A.var.index)
common = sorted(common)
adatas = [A[:, common].copy() for A in adatas]
for A in adatas:
    A.var.index = A.var.index.astype(str)

# CONCAT ALL DATA
all_data = ad.concat(adatas, join='outer', merge='same', index_unique=None)
all_data.X = csr_matrix(all_data.X)
print(all_data.obs.columns)
# Create celltype column from layer
all_data.obs['celltype'] = all_data.obs['ground_truth'] #or layer for DLPFC

# SETUP CROSS VALIDATION
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cell_indices = np.arange(all_data.n_obs)

# Store results across folds
cv_results = {
    'accuracy': [], 'f1_score': [], 'precision': [], 'recall': [],
    'ari': [], 'nmi': []
}

print(f"Starting {N_FOLDS}-fold cross-validation for cell type annotation...")

for fold, (train_val_idx, test_idx) in enumerate(kf.split(cell_indices)):
    print(f"\n=== FOLD {fold + 1}/{N_FOLDS} ===")
    
    # Split train_val into train and val
    train_idx, val_idx = np.split(
        np.random.permutation(train_val_idx),
        [int(0.8 * len(train_val_idx))]
    )
    
    # Create split labels
    split_labels = np.array([''] * all_data.n_obs, dtype=object)
    split_labels[train_idx] = 'train'
    split_labels[val_idx] = 'val'
    split_labels[test_idx] = 'test'
    all_data.obs['split'] = split_labels
    
    print(f"Fold {fold + 1}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # CONFIGURE PIPELINE
    pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()
    pipeline_config['epochs'] = 100  # You can increase this for better performance
    model_config = CellTypeAnnotationDefaultModelConfig.copy()
    
    # Encode cell types for this fold
    le = LabelEncoder()
    all_data.obs['celltype'] = le.fit_transform(all_data.obs['celltype'].astype(str))
    model_config['out_dim'] = len(le.classes_)
    
    print(f"Fold {fold + 1}: {len(le.classes_)} unique cell types")
    
    pipeline = CellTypeAnnotationPipeline(
        pretrain_prefix=PRETRAIN_VERSION,
        overwrite_config=model_config,
        pretrain_directory='ckpt',
    )
    
    # FINE-TUNE
    print(f"Fold {fold + 1}: Starting fine-tuning...")
    pipeline.fit(
        all_data,
        pipeline_config,
        split_field='split',
        train_split='train',
        valid_split='val',
        label_fields=['celltype'],
    )
    
    # INFERENCE
    print(f"Fold {fold + 1}: Running inference...")
    pipeline.predict(
        all_data,
        pipeline_config,
    )
    
    # EVALUATION
    print(f"Fold {fold + 1}: Evaluating...")
    metrics = pipeline.score(
        all_data,
        pipeline_config,
        split_field='split',
        target_split='test',
        label_fields=['celltype'],
    )
    print(f"Fold {fold + 1} metrics: {metrics}")
    
    # Store results
    cv_results['accuracy'].append(metrics['acc'])
    cv_results['f1_score'].append(metrics['f1_score'])
    cv_results['precision'].append(metrics['precision'])
    cv_results['recall'].append(metrics['recall'])


# Print cross-validation summary
print(f"\n=== CROSS-VALIDATION SUMMARY ({N_FOLDS} folds) ===")

print("\nClassification Metrics (mean , std):")
print(f"   Accuracy:  {np.mean(cv_results['accuracy']):.4f} , {np.std(cv_results['accuracy']):.4f}")
print(f"   F1 Score:  {np.mean(cv_results['f1_score']):.4f} , {np.std(cv_results['f1_score']):.4f}")
print(f"   Precision: {np.mean(cv_results['precision']):.4f} , {np.std(cv_results['precision']):.4f}")
print(f"   Recall:    {np.mean(cv_results['recall']):.4f} , {np.std(cv_results['recall']):.4f}")
