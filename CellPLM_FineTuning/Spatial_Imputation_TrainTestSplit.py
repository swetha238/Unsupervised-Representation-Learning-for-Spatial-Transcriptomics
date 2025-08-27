import warnings, sys, os
warnings.filterwarnings("ignore")

# 1) Make sure your local CellPLM clone is on-path
import numpy as np
import anndata as ad
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

from CellPLM.utils import set_seed
from CellPLM.utils.data import stratified_sample_genes_by_sparsity
from CellPLM.pipeline.imputation import (
    ImputationPipeline,
    ImputationDefaultPipelineConfig,
    ImputationDefaultModelConfig,
)

# 2) CONFIG
PRETRAIN_VERSION = '20231027_85M'
DEVICE           = 'cuda:4'
SEED             = 11
N_FOLDS          = 5  # Number of CV folds
set_seed(SEED)
dataset = 'MERFISH' # or DLPFC

# 3) LOAD ALL FILES
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

# 4) INTERSECT GENES
common = set(adatas[0].var.index)
for A in adatas[1:]:
    common &= set(A.var.index)
common = sorted(common)
adatas = [A[:, common].copy() for A in adatas]
for A in adatas:
    A.var.index = A.var.index.astype(str)

# 5) CONCAT ALL DATA
all_data = ad.concat(adatas, join='outer', merge='same', index_unique=None)

# 6) SETUP CROSS VALIDATION
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cell_indices = np.arange(all_data.n_obs)

# Store results across folds
cv_results = {
    'mse': [], 'rmse': [], 'mae': [], 'corr': [],
    'pipe_mse': [], 'pipe_rmse': [], 'pipe_mae': [], 'pipe_corr': [],
    'mean_per_gene_corr': []
}

print(f"Starting {N_FOLDS}-fold cross-validation...")

for fold, (train_val_idx, test_idx) in enumerate(kf.split(cell_indices)):
    print(f"\n=== FOLD {fold + 1}/{N_FOLDS} ===")
    
    # Split train_val into train and val
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, random_state=SEED, shuffle=True
    )
    
    # Create split labels
    split_labels = np.array([''] * all_data.n_obs, dtype=object)
    split_labels[train_idx] = 'train'
    split_labels[val_idx] = 'val'
    split_labels[test_idx] = 'test'
    all_data.obs['split'] = split_labels
    
    # 7) HOLD OUT GENES (using only train cells)
    train_data = all_data[all_data.obs['split'] == 'train']
    tg = stratified_sample_genes_by_sparsity(train_data, seed=SEED)
    tg = [str(g) for g in tg if g in all_data.var.index]
    print(f"Fold {fold + 1}: Holding out {len(tg)} genes for evaluation")
    
    # 8) STASH & ZERO
    for A in [all_data]:
        A.obsm['truth'] = A[:, tg].X.toarray()
        A[:, tg].X = 0
    
    # 9) PIPELINE CONFIG
    pconf = ImputationDefaultPipelineConfig.copy()
    pconf['epochs'] = 100  # set to your desired epochs
    mconf = ImputationDefaultModelConfig.copy()
    
    pipe = ImputationPipeline(
        pretrain_prefix    = PRETRAIN_VERSION,
        overwrite_config   = mconf,
        pretrain_directory = 'ckpt',
    )
    
    # 10) FIT (comment out for zero-shot)
    pipe.fit(
         all_data, pconf,
         split_field    ='split',
         train_split    ='train',
         valid_split    ='val',
         batch_gene_list=None,
         ensembl_auto_conversion=True
     )
    
    # 11) PREDICT ON TEST
    test_data = all_data[all_data.obs['split'] == 'test'].copy()
    test_data.var.index = test_data.var.index.astype(str)
    preds = pipe.predict(test_data, pconf)
    
    # 12) get the exact processed AnnData
    processed = pipe.common_preprocess(test_data, 0, None, ensembl_auto_conversion=False)
    print(f"Fold {fold + 1}: Processed shape: {processed.shape}  preds.shape={preds.shape}")
    
    # 13) FILTER hold-out list to genes present in processed.var.index
    filtered_tg = [g for g in tg if g in processed.var.index]
    dropped = set(tg) - set(filtered_tg)
    if dropped:
        print(f"Fold {fold + 1}: Dropped {len(dropped)} hold-out genes not in processed var.index")
    print(f"Fold {fold + 1}: Evaluating on {len(filtered_tg)} genes")
    
    # 14) SLICE & SCORE
    idxs = [processed.var.index.get_loc(g) for g in filtered_tg]
    P = preds[:, idxs]
    T = torch.tensor(test_data.obsm['truth'], dtype=torch.float32)
    truth_idxs = [tg.index(g) for g in filtered_tg]
    T = T[:, truth_idxs]
    
    # log1p + relu
    P = F.relu(torch.log1p(P))
    T = torch.log1p(T)
    
    # Debugging block
    print(f"Fold {fold + 1} DEBUG: P shape: {P.shape}, T shape: {T.shape}")
    print(f"Fold {fold + 1} DEBUG: P min/max: {P.min().item():.6f}/{P.max().item():.6f}")
    print(f"Fold {fold + 1} DEBUG: T min/max: {T.min().item():.6f}/{T.max().item():.6f}")
    print(f"Fold {fold + 1} DEBUG: Any all-zero columns in T? {(T.sum(0) == 0).any().item()}")
    
    # Remove all-zero columns
    nonzero_cols = (T.sum(0) != 0).cpu().numpy()
    T = T[:, nonzero_cols]
    P = P[:, nonzero_cols]
    filtered_tg = [g for i, g in enumerate(filtered_tg) if nonzero_cols[i]]
    
    # Update test_data for pipeline scoring
    test_data.obsm['truth'] = test_data.obsm['truth'][:, truth_idxs]
    test_data.obsm['truth'] = test_data.obsm['truth'][:, nonzero_cols]
    
    # Manual metrics
    y_p = P.flatten().cpu().numpy()
    y_t = T.flatten().cpu().numpy()
    
    mse  = mean_squared_error(y_t, y_p)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_t, y_p)
    corr = np.corrcoef(y_t, y_p)[0,1]
    
    print(f"Fold {fold + 1} Manual metrics:")
    print(f"   MSE:  {mse:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   Corr: {corr:.4f}")
    
    # Pipeline metrics
    metrics = pipe.score(
        test_data,
        evaluation_config = {'target_genes': filtered_tg},
        label_fields      = ['truth'],
    )
    print(f"Fold {fold + 1} Pipeline metrics: {metrics}")
    
    # Per-gene correlation
    per_gene_corrs = [
        np.corrcoef(T[:, i].cpu().numpy(), P[:, i].cpu().numpy())[0, 1]
        for i in range(T.shape[1])
    ]
    mean_per_gene_corr = np.nanmean(per_gene_corrs)
    print(f"Fold {fold + 1} Mean per-gene correlation: {mean_per_gene_corr:.4f}")
    
    # Store results
    cv_results['mse'].append(mse)
    cv_results['rmse'].append(rmse)
    cv_results['mae'].append(mae)
    cv_results['corr'].append(corr)
    cv_results['pipe_mse'].append(metrics['mse'])
    cv_results['pipe_rmse'].append(metrics['rmse'])
    cv_results['pipe_mae'].append(metrics['mae'])
    cv_results['pipe_corr'].append(metrics['corr'])
    cv_results['mean_per_gene_corr'].append(mean_per_gene_corr)

# Print cross-validation summary
print(f"\n=== CROSS-VALIDATION SUMMARY ({N_FOLDS} folds) ===")

print("\nManual Metrics (mean , std):")
print(f"   MSE:  {np.mean(cv_results['mse']):.6f} , {np.std(cv_results['mse']):.6f}")
print(f"   RMSE: {np.mean(cv_results['rmse']):.6f} , {np.std(cv_results['rmse']):.6f}")
print(f"   MAE:  {np.mean(cv_results['mae']):.6f} , {np.std(cv_results['mae']):.6f}")
print(f"   Corr: {np.mean(cv_results['corr']):.4f} , {np.std(cv_results['corr']):.4f}")

print("\nPipeline Metrics (mean , std):")
print(f"   MSE:  {np.mean(cv_results['pipe_mse']):.6f} , {np.std(cv_results['pipe_mse']):.6f}")
print(f"   RMSE: {np.mean(cv_results['pipe_rmse']):.6f} , {np.std(cv_results['pipe_rmse']):.6f}")
print(f"   MAE:  {np.mean(cv_results['pipe_mae']):.6f} , {np.std(cv_results['pipe_mae']):.6f}")
print(f"   Corr: {np.mean(cv_results['pipe_corr']):.4f} , {np.std(cv_results['pipe_corr']):.4f}")

print(f"\nMean Per-Gene Correlation: {np.mean(cv_results['mean_per_gene_corr']):.4f} , {np.std(cv_results['mean_per_gene_corr']):.4f}")

# Save results to file
import json
with open('cv_results.json', 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    json_results = {k: [float(x) for x in v] for k, v in cv_results.items()}
    json.dump(json_results, f, indent=2)
print("\nResults saved to cv_results.json")