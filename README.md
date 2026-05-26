TriNCFA-Net
Official implementation of TriNCFA-Net: A Lightweight Tri-Modal Network with Noise-Contrastive Feature Alignment for Robust and Efficient Automatic Modulation Recognition.
TriNCFA-Net is a lightweight automatic modulation recognition (AMR) framework. It combines three complementary signal representations—raw I/Q sequences, amplitude-phase differential features, and Gramian Angular Field (GAF) representations—and introduces a statistic-driven hierarchical gated fusion (SHGF) module and a noise-contrastive feature alignment (NCFA) training strategy.
Highlights
Tri-modal signal representation: raw I/Q, amplitude-phase differential features, and GAF-based temporal correlation features.
Statistic-driven hierarchical gated fusion (SHGF): adaptively fuses auxiliary modalities with the primary I/Q modality according to signal-quality-related statistics.
Noise-contrastive feature alignment (NCFA): improves low-SNR robustness through supervised contrastive learning and residual feature reconstruction.
Lightweight deployment: designed for a favorable accuracy-complexity trade-off in AMR tasks.
Unified training and evaluation pipeline: supports TriNCFA-Net and several baseline models on RadioML datasets.
Repository Structure
```text
TriNCFA-Net/
├── models/                         # Model implementations
│   ├── TriNCFANet/                 # Proposed TriNCFA-Net
│   ├── AMC_Net/                    # AMCNet baseline
│   ├── MCLDNN/                     # MCLDNN baseline
│   ├── PETCGDNN/                   # PET-CGDNN baseline
│   ├── FEA_T/                      # FEA-T baseline
│   ├── MCFormer/                   # MCFormer baseline
│   ├── IQFormer/                   # IQFormer baseline
│   └── SMT/                        # SMT baseline
├── utils/                          # Utility functions, if used by model modules
├── dataset/                        # Place downloaded datasets here
├── weights/                        # Saved model checkpoints
├── results/                        # Evaluation metrics, plots, and summaries
├── train.py                        # Train/evaluate one model on one dataset
├── train_all.py                    # Batch training and result aggregation
├── plot_snr_comparison.py          # Plot SNR-accuracy comparison curves
├── viz_metrics.py                  # Visualization and metric utilities
└── README.md
```
Environment
The code is based on Python and PyTorch. A recommended environment is:
```bash
conda create -n trincfa python=3.9 -y
conda activate trincfa
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn matplotlib tqdm h5py thop
```
If your CUDA version is different, install PyTorch following the official PyTorch instructions.
Datasets
This repository supports the following public AMR datasets:
Dataset option	Expected file path	Signal length
`RML2016a`	`dataset/RML2016.10a_dict.pkl`	128
`RML2016b`	`dataset/RML2016.10b.dat`	128
`RML2018a`	`dataset/GOLD_XYZ_OSC.0001_1024.hdf5`	1024
Please download the datasets separately and place them under the `dataset/` directory.
Example:
```text
dataset/
├── RML2016.10a_dict.pkl
├── RML2016.10b.dat
└── GOLD_XYZ_OSC.0001_1024.hdf5
```
For `RML2018a`, the full dataset can be large. The training script supports sub-sampling by `--samples_per_class_snr`.
Quick Start
1. Train TriNCFA-Net on RML2016.10a
```bash
python train.py \
  --dataset RML2016a \
  --model_name TriNCFA_Net \
  --epochs 100 \
  --batch_size 128 \
  --lr 1e-3
```
If `--dataset_path` is not specified, the script uses the default path:
```text
dataset/RML2016.10a_dict.pkl
```
2. Train on RML2016.10b
```bash
python train.py \
  --dataset RML2016b \
  --model_name TriNCFA_Net \
  --epochs 100 \
  --batch_size 128 \
  --lr 1e-3
```
Default dataset path:
```text
dataset/RML2016.10b.dat
```
3. Train on RML2018.01a
```bash
python train.py \
  --dataset RML2018a \
  --model_name TriNCFA_Net \
  --epochs 80 \
  --batch_size 128 \
  --lr 1e-3 \
  --samples_per_class_snr 512
```
Default dataset path:
```text
dataset/GOLD_XYZ_OSC.0001_1024.hdf5
```
If you want to use a custom dataset location:
```bash
python train.py \
  --dataset RML2018a \
  --dataset_path /path/to/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --model_name TriNCFA_Net
```
Supported Models
The following model names can be used with `--model_name`:
```text
AMCNet
MCLDNN
PETCGDNN
FEA_T
MCFormer
IQFormer
SMT
TriNCFA_Net
```
Example:
```bash
python train.py --dataset RML2016a --model_name MCLDNN --epochs 100
```
Outputs
After training and evaluation, the script automatically creates:
```text
weights/
└── <dataset>_<model>_baseline_best.pth

results/
├── <dataset>_<model>_baseline_best.json
├── <dataset>_<model>_snr_curve.png
├── <dataset>_<model>_snr_curve.pdf
├── <dataset>_<model>_confusion_snr+10dB.png
├── <dataset>_<model>_confusion_snr+10dB.pdf
├── <dataset>_<model>_tsne_snr*.png
└── <dataset>_<model>_tsne_snr*.pdf
```
The JSON file records overall accuracy, F1 score, low-SNR accuracy, high-SNR accuracy, parameter count, FLOPs, inference latency, and SNR-wise accuracy.
Batch Training and Comparison
To train all supported models on one dataset:
```bash
python train_all.py \
  --dataset RML2016a \
  --epochs 100 \
  --batch_size 128
```
To train only selected models:
```bash
python train_all.py \
  --dataset RML2016a \
  --models TriNCFA_Net MCLDNN IQFormer SMT \
  --epochs 100
```
To skip some models:
```bash
python train_all.py \
  --dataset RML2018a \
  --epochs 80 \
  --skip MCFormer FEA_T
```
To force retraining even if result JSON files already exist:
```bash
python train_all.py \
  --dataset RML2016a \
  --epochs 100 \
  --retrain
```
The batch script generates summary files:
```text
results/
├── <dataset>_summary.json
├── <dataset>_summary.md
├── <dataset>_summary.csv
└── <dataset>_snr_compare.png/.pdf
```
Plot SNR-Accuracy Comparison
If result JSON files already exist, you can generate SNR comparison curves directly:
```bash
python plot_snr_comparison.py \
  --dataset RML2016a \
  --highlight TriNCFA_Net
```
For RML2018.01a:
```bash
python plot_snr_comparison.py \
  --dataset RML2018a \
  --highlight TriNCFA_Net \
  --output results/RML2018a_snr_compare.png
```
Reproducing the Main Experiments
A typical reproduction workflow is:
```bash
# RML2016.10a
python train_all.py --dataset RML2016a --epochs 100 --batch_size 128

# RML2016.10b
python train_all.py --dataset RML2016b --epochs 100 --batch_size 128

# RML2018.01a
python train_all.py --dataset RML2018a --epochs 80 --batch_size 128 --samples_per_class_snr 512
```
Then generate or update SNR comparison plots:
```bash
python plot_snr_comparison.py --dataset RML2016a
python plot_snr_comparison.py --dataset RML2016b
python plot_snr_comparison.py --dataset RML2018a
```
Notes
The scripts automatically split each dataset into training, validation, and testing subsets using stratified sampling.
For `RML2016a` and `RML2016b`, the default signal length is 128.
For `RML2018a`, the default signal length is 1024.
For fair comparison, use the same random seed, training epochs, and data split settings across models.
If GPU memory is insufficient on `RML2018a`, reduce `--batch_size` or `--samples_per_class_snr`.
Citation
If this repository is helpful to your research, please consider citing our work:
```bibtex
@article{ma2026trincfanet,
  title   = {TriNCFA-Net: A Lightweight Tri-Modal Network with Noise-Contrastive Feature Alignment for Robust and Efficient Automatic Modulation Recognition},
  author  = {Ma, Ruipeng and Wu, Di and Hu, Tao and Li, Tingli and Han, Yi},
  journal = {IEEE Journal/Transactions},
  year    = {2026}
}
```
Contact
For questions or suggestions, please contact:
```text
Ruipeng Ma: MRP_1018@163.com
```
License
Please add a license file before public release if you want to specify how others may use or redistribute this code.
