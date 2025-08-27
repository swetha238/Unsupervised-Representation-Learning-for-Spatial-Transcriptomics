# CellPLM Fine-tuning Experiments for Spatial Transcriptomics

This repository contains the implementation of fine-tuning experiments for CellPLM (Cell Pre-trained Language Model) on various spatial transcriptomics datasets. The experiments focus on three main downstream tasks: cell type annotation, cell embedding generation, and spatial imputation.

## Project Overview

This research implements and evaluates CellPLM's performance on multiple spatial transcriptomics datasets through systematic fine-tuning experiments. The work demonstrates the model's capability to adapt to diverse tissue types and experimental conditions, providing insights into its generalizability and effectiveness for spatial transcriptomics analysis.

## Repository Structure

```
README.md                                       # This documentation file
cellplm-finetuning/
├── Cell_annotation_BaselineComparison.py       # Cell type annotation baseline experiment
├── Cell_Annotation_TrainTestSplit.py           # Cell annotation with cross-validation
├── Cell_embedding.py                           # Cell embedding generation experiment
├── Spatial_Imputation_BaselineComparison.py    # Spatial imputation baseline experiment
├── Spatial_Imputation_TrainTestSplit.py        # Spatial imputation with cross-validation
├── ckpt/                                       # Model checkpoints directory
│   ├── 20231027_85M.best.ckpt                 # Pre-trained model weights
│   └── 20231027_85M.config.json              # Model configuration
└── data/                                       # Dataset directory 
    ├── MERFISH_1.h5ad to MERFISH_6.h5ad       # MERFISH datasets
    ├── sample_1.h5ad to sample_12.h5ad        # DLPFC datasets
   
```

## Experimental Setup

### Prerequisites

1. **CellPLM Installation**: Ensure CellPLM is properly installed and accessible
2. **Python Environment**: Python 3.8+ with required packages
3. **GPU Access**: CUDA-compatible GPU for model training and inference
4. **Data Preparation**: All datasets must be in AnnData (.h5ad) format

### Required Dependencies

```bash
1. pip install numpy scipy pandas anndata scanpy torch scikit-learn matplotlib seaborn h5py hdf5plugin

2. pip install cellplm
```

### Model Checkpoints

The experiments require the pre-trained CellPLM model checkpoint:
- **Checkpoint Path**: `ckpt/20231027_85M.best.ckpt` or `ckpt/20230926_85M.best.ckpt`
- **Config Path**: `ckpt/20231027_85M.config.json` or `ckpt/20230926_85M.config.json`
- **Model Version**: 20231027_85M (85M parameter model)

## Dataset Configuration

### Supported Datasets

#### 1. MERFISH Datasets (Spatial Transcriptomics)
- **Files**: `MERFISH_1.h5ad` to `MERFISH_6.h5ad`
- **Tissue Type**: Mouse brain tissue
- **Technology**: Multiplexed Error-Robust FISH (MERFISH)
- **Usage**: Set `dataset = 'MERFISH'` in experiment scripts

#### 2. DLPFC Datasets (Brain Tissue)
- **Files**: `sample_1.h5ad` to `sample_12.h5ad`
- **Tissue Type**: Human dorsolateral prefrontal cortex
- **Technology**: 10x Visium spatial transcriptomics
- **Usage**: Set `dataset = 'DLPFC'` in experiment scripts

#### 3. Lung Cancer Datasets
- **Query**: `HumanLungCancerPatient2_filtered_ensg.h5ad`
- **Reference**: `GSE131907_Lung_ensg.h5ad`
- **Tissue Type**: Human lung cancer
- **Usage**: Set `DATASET = 'Lung'` in experiment scripts

#### 4. Liver Cancer Datasets
- **Query**: `HumanLiverCancerPatient2_filtered_ensg.h5ad`
- **Reference**: `GSE151530_Liver_ensg.h5ad`
- **Tissue Type**: Human liver cancer
- **Usage**: Set `DATASET = 'Liver'` in experiment scripts

#### 5. Additional Datasets
- **GSE155468**: `gse155468.h5ad`
- **hPancreas**: `demo_train.h5ad`, `demo_test.h5ad`
- **MS**: `c_data.h5ad`, `filtered_ms_adata.h5ad` (Multiple sclerosis)

## Dataset Sources and Downloads

### Dataset Links
All datasets required for running the experiments are available from the official [CellPLM GitHub repository](https://github.com/OmicsML/CellPLM/tree/main/data).

**Additional Direct Links**:
MERFISH: http://sdmbench.drai.cn/
DLPFC: http://sdmbench.drai.cn/ (Datasets named as 10x visium)

**Required Files**:
- `HumanLungCancerPatient2_filtered_ensg.h5ad`, `GSE131907_Lung_ensg.h5ad` (Human lung cancer data)
- `HumanLiverCancerPatient2_filtered_ensg.h5ad`, `GSE151530_Liver_ensg.h5ad` (Human liver cancer data)
- `gse155468.h5ad` (Lung tissue single-cell data)
- `demo_train.h5ad`, `demo_test.h5ad` (Human pancreas single-cell data)
- `c_data.h5ad`, `filtered_ms_adata.h5ad` (Multiple sclerosis tissue data)



## Experiment Scripts

### 1. Cell Type Annotation Experiments

#### `Cell_annotation_BaselineComparison.py`
**Purpose**: Baseline Comparison evaluation of CellPLM for cell type annotation tasks.

**Configuration**:
```python
PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:3'
DATASET = 'hPancreas'  # Options: 'hPancreas', 'MS'
```

**Key Features**:
- Fine-tunes CellPLM on cell type annotation task
- Evaluates classification performance using accuracy, F1-score, precision, and recall
- Generates confusion matrices and UMAP visualizations
- Supports multiple sclerosis and pancreas datasets

**Usage**:
```bash
python Cell_annotation_BaselineComparison.py
```

#### `Cell_Annotation_TrainTestSplit.py`
**Purpose**: Cross-validation evaluation of cell type annotation performance.

**Configuration**:
```python
PRETRAIN_VERSION = '20230926_85M'
DEVICE = 'cuda:3'
SEED = 42
N_FOLDS = 5
dataset = 'MERFISH'  # Options: 'MERFISH', 'DLPFC'
```

**Key Features**:
- Implements K-fold cross-validation (5 folds)
- Handles multiple datasets with gene intersection
- Stratified sampling for balanced evaluation
- Comprehensive performance metrics across folds

**Usage**:
```bash
python Cell_Annotation_TrainTestSplit.py
```

### 2. Cell Embedding Generation

#### `Cell_embedding.py`
**Purpose**: Generation and evaluation of cell embeddings using CellPLM.

**Configuration**:
```python
PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:1'
dataset = 'GSE155468'  # Options: 'GSE155468', 'DLPFC'
```

**Key Features**:
- Generates high-dimensional cell embeddings
- Performs UMAP dimensionality reduction and visualization
- Evaluates clustering performance using Leiden algorithm
- Supports multiple tissue types and experimental conditions

**Usage**:
```bash
python Cell_embedding.py
```

### 3. Spatial Imputation Experiments

#### `Spatial_Imputation_BaselineComparison.py`
**Purpose**: Baseline evaluation of spatial imputation performance.

**Configuration**:
```python
DATASET = 'Liver'  # Options: 'Lung', 'Liver'
PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:4'
```

**Key Features**:
- Implements gene hold-out strategy for evaluation
- Combines spatial and single-cell RNA-seq data
- Evaluates imputation accuracy using multiple metrics
- Supports cross-tissue generalization

**Usage**:
```bash
python Spatial_Imputation_BaselineComparison.py
```

#### `Spatial_Imputation_TrainTestSplit.py`
**Purpose**: Cross-validation evaluation of spatial imputation performance.

**Configuration**:
```python
PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:4'
SEED = 11
N_FOLDS = 5
dataset = 'MERFISH'  # Options: 'MERFISH', 'DLPFC'
```

**Key Features**:
- K-fold cross-validation for robust evaluation
- Gene-based hold-out strategy using stratified sampling
- Comprehensive evaluation metrics (MSE, RMSE, MAE, correlation)
- Per-gene correlation analysis

**Usage**:
```bash
python Spatial_Imputation_TrainTestSplit.py
```

## Experimental Parameters

### Model Configuration
- **Pre-trained Model**: CellPLM 85M parameter model
- **Fine-tuning Epochs**: 100-600 (dataset-dependent)
- **Batch Size**: Optimized for GPU memory constraints
- **Learning Rate**: Default CellPLM settings

### Evaluation Metrics

#### Cell Type Annotation
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1-score for imbalanced classes
- **Precision**: Positive predictive value
- **Recall**: Sensitivity or true positive rate
- **Adjusted Rand Index (ARI)**: Clustering similarity
- **Normalized Mutual Information (NMI)**: Information-theoretic similarity

#### Spatial Imputation
- **Mean Squared Error (MSE)**: Average squared prediction error
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Correlation Coefficient**: Pearson correlation between predicted and true values
- **Per-gene Correlation**: Individual gene-level performance

#### Cell Embedding
- **Clustering Metrics**: ARI, NMI for cell type clustering
- **Visualization Quality**: UMAP embedding coherence
- **Computational Efficiency**: Training and inference time

## Data Preprocessing

### Standard Pipeline
1. **Data Loading**: AnnData format with gene expression matrices
2. **Quality Control**: Cell and gene filtering based on expression thresholds
3. **Normalization**: Log-transformation and scaling
4. **Gene Intersection**: Common gene identification across datasets
5. **Sparse Matrix Conversion**: Memory-efficient storage
6. **Metadata Alignment**: Cell type labels and spatial coordinates


## Reproducibility


### Hardware Requirements
- **GPU**: CUDA-compatible GPU (V100 or equivalent recommended)
- **Memory**: 8GB+ GPU memory for training
- **Storage**: Sufficient space for datasets and results


## Troubleshooting

### Common Issues

1. **Missing Checkpoint Files**
   ```
   FileNotFoundError: ckpt/20231027_85M.best.ckpt not found
   ```
   **Solution**: Ensure checkpoint files are in the `ckpt/` directory

2. **CUDA Memory Errors**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size or use CPU: `DEVICE = 'cpu'`

3. **Missing Dataset Files**
   ```
   FileNotFoundError: Missing file: MERFISH_1.h5ad
   ```
   **Solution**: Place all required datasets in the `data/` directory

4. **CellPLM Import Errors**
   ```
   ModuleNotFoundError: No module named 'CellPLM'
   ```
   **Solution**: Ensure CellPLM is properly installed and in Python path


## Citation

If you use this code in your research, please cite:

```bibtex
@article{cellplm2024,
  title={CellPLM: Pre-trained Language Model for Single-cell Transcriptomics},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

