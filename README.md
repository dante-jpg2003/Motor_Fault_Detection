# Motor Fault Detection — CNN-GRU Fusion Model

A deep learning pipeline for synchronous motor electrical fault classification using a hybrid CNN-GRU architecture. Built as a self-defined project for CMPE 401.

The model combines three one-dimensional convolutional blocks for spatial feature extraction with a two-layer Gated Recurrent Unit (GRU) for temporal modelling, evaluated across five window sizes (100, 200, 400, 600, 800 time points) on a public synchronous motor fault benchmark dataset.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Running on Google Colab](#running-on-google-colab)
- [Results](#results)
- [References](#references)

---

## Dataset

This project uses the public synchronous motor electrical fault dataset from:

> Z. Sun, R. Machlev, Q. Wang, J. Belikov, Y. Levron, and D. Baimel,
> "A public data-set for synchronous motor electrical faults diagnosis with CNN and LSTM reference classifiers,"
> *Energy and AI*, vol. 14, p. 100274, 2023.

Download the `.mat` files from:
https://gitlab.com/power-systems-technion/motor-faults/-/tree/main

Place all six `.mat` files in `data/raw/` before running any scripts:

```
data/raw/
├── Preprocessed_No_failed.mat
├── Preprocessed_Disconnect_Phase_10_11_21_.mat
├── Preprocessed_Short_between_two_phases_.mat
├── Preprocessed_Test_Data_Short_phases_Ln_G_.mat
├── Preprocessed_Rotor_Current_Failed_R_.mat
└── Preprocessed_Test_Data_Rotor_Current_Faild.mat
```

The dataset contains six classes across 3,392 experiments, each with 10,000 time points and 9 signal channels:

| Label | Class | Description |
|-------|-------|-------------|
| 0 | VREC | Variation of rotor excitation current |
| 1 | OP | Open phase between inverter and motor |
| 2 | REVD | Rotor excitation voltage disconnection |
| 3 | 2PSC | Two-phase short circuit |
| 4 | 1PSC | One phase-to-neutral short circuit |
| 5 | NF | Normal operation (no fault) |

---

## Project Structure

```
Motor_Fault_Detection/
│
├── data/
│   └── raw/                    # .mat files (not tracked by git — download separately)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   └── 02_training.ipynb       # Colab training and evaluation notebook
│
├── scripts/
│   ├── dataset.py              # Data loading, preprocessing, windowing, Dataset class
│   ├── model.py                # CNN-GRU architecture
│   ├── train.py                # Training loop with weighted loss and LR scheduler
│   └── evaluate.py             # Evaluation metrics, confusion matrix, per-class F1
│
├── configs/
│   └── eda_config.json         # Channel map and dataset constants from EDA
│
├── results/
│   ├── figures/                # Saved plots (confusion matrix, training curves, etc.)
│   ├── best_model.pt           # Saved model checkpoint
│   ├── training_history.json   # Per-epoch loss and accuracy
│   └── evaluation_results.json # Final evaluation metrics
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Requirements

- Python 3.11+
- A virtual environment is used for this project (see [Setup](#setup))

### Python Libraries

```
torch>=2.0
scipy
numpy
scikit-learn
matplotlib
seaborn
pyyaml
tqdm
jupyter
ipykernel
tensorboard
```

## Setup

This project uses a Python virtual environment to isolate dependencies. Follow these steps to get started locally.

**Step 1 — Clone the repository**

```bash
git clone https://github.com/dante-jpg2003/Motor_Fault_Detection.git
cd Motor_Fault_Detection
```

**Step 2 — Create and activate a virtual environment**

```bash
# Create the virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

You will see `(venv)` at the start of your terminal prompt when the environment is active.

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Download the dataset**

Download the `.mat` files from the GitLab link above and place them in `data/raw/`.

**Step 5 — Select the venv as your kernel in VSCode**

Press `Ctrl+Shift+P` → `Python: Select Interpreter` → choose the venv interpreter. This ensures notebooks and scripts use the correct environment.

---

## Usage

All scripts are run from the project root with the virtual environment activated.

### 1. Exploratory Data Analysis

Open and run `notebooks/01_eda.ipynb` in VSCode or Jupyter. This notebook covers:

- Loading and inspecting the `.mat` files
- Class distribution analysis
- Per-channel signal statistics and visualisation
- Fault signal comparison across all six classes
- Windowing strategy analysis

### 2. Verify the Dataset Pipeline

```bash
python scripts/dataset.py
```

Loads all `.mat` files, performs the train/test split, normalises the data, and creates windowed datasets. Prints a batch shape verification at the end — expected output is `(32, 9, 800)`.

### 3. Verify the Model Architecture

```bash
python scripts/model.py
```

Runs a forward pass with dummy inputs for all five window sizes and prints the output shape and parameter count. Expected output shape is `(32, 6)` for all window sizes.

### 4. Train the Model (Local — Small Config)

For local development and testing on CPU:

```bash
python scripts/train.py
```

The default config in `train.py` uses `window_size=800`, `batch_size=64`, and `epochs=5`. This is intended for verifying the pipeline locally. Full training should be run on Colab with GPU.

### 5. Evaluate a Trained Model

```bash
python scripts/evaluate.py
```

Loads the saved checkpoint from `results/best_model.pt`, runs inference on the test set, and outputs:

- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix (counts and normalised)
- Saved figures in `results/figures/`

---

## Running on Google Colab

Full training and the window size experiments are designed to run on Google Colab with a T4 GPU. Before running, ensure your Google Drive has the following structure:

```
My Drive/
└── Motor_Fault_Detection/
    ├── data/
    │   └── raw/        ← upload .mat files here
    └── results/        ← training outputs saved here
```

Open `notebooks/02_training.ipynb` in Colab and run the cells in order:

1. **Cell 1** — Clone repo from GitHub and mount Google Drive
2. **Cell 2** — Install dependencies (`scipy`, `scikit-learn`)
3. **Cell 3** — Verify GPU is available (`Runtime → Change runtime type → T4 GPU`)
4. **Cell 4** — Set environment variables for data and results paths
5. **Cell 5** — Run full training (30 epochs, window=800)
6. **Cell 6** — Plot training curves
7. **Cell 7** — Run evaluation and generate confusion matrix
8. **Cell 8** — Window size experiments (trains and evaluates all five window sizes)
9. **Cell 9** — Window size comparison plot
10. **Cell 10** — Channel ablation study

Results and figures are saved directly to Google Drive and persist between Colab sessions.

> **Note:** Colab sessions reset when disconnected. Re-run Cell 1 at the start of each session to re-clone the repo. Data on Drive is permanent and does not need to be re-uploaded.

---

## Results

The CNN-GRU model achieves the following performance on the test set (window size = 800, stride = 800):

| Metric | Score |
|--------|-------|
| Overall Accuracy | 100% |
| Macro Precision | 1.000 |
| Macro Recall | 1.000 |
| Macro F1-Score | 1.000 |

Performance is consistent across all five window sizes tested (100 to 800 time points).

**Comparison with paper baselines (CNN, window = 800 points):**

| Model | Feature | Accuracy |
|-------|---------|----------|
| Paper CNN | Stator voltage | 97.05% |
| Paper CNN | Stator current | 98.38% |
| Paper CNN | Rotor current | 98.97% |
| Paper CNN | Rotor speed | 43.81% |
| **CNN-GRU (this work)** | **All 9 channels** | **100.00%** |

> **Important note on comparison:** The paper's pipeline centres windows around the fault onset time point, making it a fault transition detection task. This implementation uses full recordings without fault-onset centring, making it a steady-state fault classification task. Direct numerical comparison should be interpreted with this distinction in mind. See the channel ablation study results in `results/` for a full discussion.

---

## References

Sun, Z., Machlev, R., Wang, Q., Belikov, J., Levron, Y., & Baimel, D. (2023). A public data-set for synchronous motor electrical faults diagnosis with CNN and LSTM reference classifiers. *Energy and AI*, 14, 100274. https://doi.org/10.1016/j.egyai.2023.100274

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
