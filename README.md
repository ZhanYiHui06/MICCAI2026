# SC-UNSB

**SC-UNSB (Unpaired Neural Schrödinger Bridge for Cell Staining Style Transfer)** - A PyTorch implementation of cell staining style transfer tool based on the Neural Schrödinger Bridge method.

## Key Features

- **Multi-step Refinement**: Progressive image translation through Neural Function Evaluations (NFE)
- **Dense Normalization**: Improved normalization method to enhance cell staining transfer quality
- **Background Detection**: Prevents generation of false cells in blank regions
- **Stratified Sampling**: V2 version supports stratified sampling strategy for background/dense cell regions

## Installation

```bash
# Create environment
conda create -n sc_unsb python=3.8
conda activate sc_unsb

# Install dependencies
pip install -e .
```

## Quick Start

### Training

```bash
# V1 training (resize-based)
python scripts/train.py --config scripts/configs/v1_cell.yaml

# V2 training (patch-based)
python scripts/train.py --config scripts/configs/v2_cell.yaml

# SC-UNSB training (with Dense Normalization)
python scripts/train.py --config scripts/configs/sc_unsb.yaml
```

### Inference

```bash
# Test the model
python scripts/test.py --name cell_sc_unsb --dataroot datasets/cell \
    --dataset_mode cell_dataset_v2 --phase test
```

## Dataset Preparation

Datasets should be organized in the following structure:

```
datasets/
└── [dataset_name]/
    ├── trainA/    # Source domain training images
    ├── trainB/    # Target domain training images
    ├── testA/     # Source domain test images
    └── testB/     # Target domain test images
```

For detailed instructions, please refer to: `datasets/README.txt`

### Evaluation Dataset

We have reconstructed a new evaluation dataset based on the CRIC Dataset and Comparison Detector Dataset, specifically designed to assess the cross-domain performance of cervical screening models.

* Download Link: [OneDrive](https://5ltfb6-my.sharepoint.com/:u:/g/personal/zyh_nexusoff_onmicrosoft_com/IQA5gkC4JlaQQJVKEESjK3h7AU_g9E4_0xLqdCWCMQpKr4M?e=8XEjQQ)

## Evaluation

Use the evaluation script to calculate FID, KID, and PSNR metrics:

```bash
# Calculate FID and KID
python scripts/evaluate.py --real datasets/cell/testB --fake results/cell_sc_unsb/test_latest/images

# Also calculate PSNR (requires paired images)
python scripts/evaluate.py --real datasets/cell/testB --fake results/cell_sc_unsb/test_latest/images --psnr
```

## Project Structure

```
SC-UNSB/
├── sc_unsb/           # Core code package
│   ├── models/        # Model implementations
│   ├── data/          # Dataset classes
│   ├── options/       # Command-line options
│   └── utils/         # Utility functions
├── scripts/           # Training/inference scripts
│   ├── train.py
│   ├── test.py
│   └── configs/       # YAML configuration files
├── tools/             # Data preparation tools
└── docs/              # Documentation
```

## Citation

If you use this project, please cite the original UNSB paper:

```bibtex
@InProceedings{kim2023unsb,
  title={Unpaired Image-to-Image Translation via Neural Schrödinger Bridge},
  author={Beomsu Kim and Gihyun Kwon and Kwanyoung Kim and Jong Chul Ye},
  booktitle={ICLR},
  year={2024}
}
```

## License

This project is based on the original UNSB project, focusing on cell staining style transfer applications.
