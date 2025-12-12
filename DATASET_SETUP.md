# Chest CT Segmentation Dataset Setup

## ✅ Dataset Downloaded Successfully

The **Chest CT Segmentation** dataset from Kaggle has been downloaded and organized.

### Dataset Details
- **Source**: Kaggle (`polomarco/chest-ct-segmentation`)
- **Location**: `/Users/edwin/Desktop/Business/Technological/fl-sam2-lora/dataset/chest-ct-segmentation`
- **Images**: 17,011 files
- **Masks**: 16,708 files
- **Metadata**: `train.csv` (contains patient IDs and split information)

### Directory Structure
```
dataset/chest-ct-segmentation/
├── images/
│   └── images/
│       ├── ID00007637202177411956430_0.jpg
│       ├── ID00007637202177411956430_1.jpg
│       └── ... (17,011 total)
├── masks/
│   └── masks/
│       ├── ID00007637202177411956430_mask_0.jpg
│       ├── ID00007637202177411956430_mask_1.jpg
│       └── ... (16,708 total)
└── train.csv
```

### Notebook Updates
The `notebooks/local.ipynb` has been updated with the correct paths:
- `PROJECT_ROOT`: `/Users/edwin/Desktop/Business/Technological/fl-sam2-lora`
- `TASK_MODULE_PATH`: `PROJECT_ROOT / "fl-sam2-lora" / "fl-sam2-lora"`
- `SAM2_LIB_PATH`: `PROJECT_ROOT / "segment-anything-2"`
- `DATASET_PATH`: `PROJECT_ROOT / "dataset" / "chest-ct-segmentation"`

### Usage
The dataset is ready to use in the federated learning pipeline. The notebook `notebooks/local.ipynb` will automatically detect and use this real dataset instead of synthetic data.

### Download Script
A script `download_dataset.py` is available for re-downloading or updating the dataset if needed. It uses the Kaggle API token stored in the script.

### Next Steps
1. Open `notebooks/local.ipynb` in Jupyter
2. Run the cells to start the federated learning pipeline
3. The notebook will automatically use the real dataset

