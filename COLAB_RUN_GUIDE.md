# Google Colab Run Guide

This document explains how to run this project in Google Colab without modifying the existing project files.

## Recommended Folder Layout

Upload the whole project to Google Drive with this structure:

```text
MyDrive/ariel-inverse-modeling/
  competitive_solution.py
  utils.py
  hackathon_starter_solution_groningen.ipynb
  Hackathon_training/
    Training_SpectralData.hdf5
    Test_SpectralData.hdf5
    Training_targets.csv
    Training_supplementary_data.csv
    Training_supp_simulation_data.csv
    Test_supplementary_data.csv
  img/
```

The important part is that both HDF5 files and all CSV files are inside `Hackathon_training/`.

## Option 1: Run the Starter Notebook

Use this if you want the simplest baseline first.

### Step 1: Open Colab and enable GPU

In Colab:

1. Open a new notebook.
2. Go to `Runtime` -> `Change runtime type`.
3. Set `Hardware accelerator` to `GPU`.

### Step 2: Mount Google Drive

Run this in the first cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Move into the project directory

```python
%cd /content/drive/MyDrive/ariel-inverse-modeling
```

### Step 4: Verify files exist

```python
import os

paths = [
    'hackathon_starter_solution_groningen.ipynb',
    'utils.py',
    'Hackathon_training/Training_SpectralData.hdf5',
    'Hackathon_training/Test_SpectralData.hdf5',
    'Hackathon_training/Training_targets.csv',
    'Hackathon_training/Training_supplementary_data.csv',
    'Hackathon_training/Training_supp_simulation_data.csv',
    'Hackathon_training/Test_supplementary_data.csv',
]

for path in paths:
    print(path, os.path.exists(path))
```

If any value is `False`, fix the Drive folder layout before continuing.

### Step 5: Install dependencies

```python
!pip install -q numpy pandas scipy scikit-learn h5py matplotlib
```

### Step 6: Open the project notebook

Open `hackathon_starter_solution_groningen.ipynb` from Google Drive using Colab.

Before running its cells, add one code cell at the top:

```python
%cd /content/drive/MyDrive/ariel-inverse-modeling
```

Then run the notebook from top to bottom.

## Option 2: Run the Competitive Script

Use this if you want the stronger neural-network pipeline.

### Step 1: Mount Drive and enter the project folder

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive/ariel-inverse-modeling
```

### Step 2: Install dependencies

```python
!pip install -q numpy pandas scipy scikit-learn h5py matplotlib
```

### Step 3: Check PyTorch and GPU

```python
import torch
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
```

### Step 4: Patch a Colab-friendly copy

This creates a separate file for Colab so the original `competitive_solution.py` stays unchanged.

```python
from pathlib import Path

src = Path('competitive_solution.py')
dst = Path('competitive_solution_colab.py')
text = src.read_text()

text = text.replace('N_SPLITS         = 5', 'N_SPLITS         = 3')
text = text.replace('PRETRAIN_EPOCHS  = 60', 'PRETRAIN_EPOCHS  = 10')
text = text.replace('FINETUNE_EPOCHS  = 120', 'FINETUNE_EPOCHS  = 20')
text = text.replace('N_MC_SAMPLES     = 50', 'N_MC_SAMPLES     = 10')
text = text.replace(
    'DEVICE           = torch.device("cpu")   # change to "cuda" if GPU available',
    'DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
)

dst.write_text(text)
print('Created competitive_solution_colab.py')
```

This version is smaller and more realistic for a free Colab session.

### Step 5: Run the Colab copy

```python
!python competitive_solution_colab.py
```

When it finishes, it should generate:

```text
lb_mu_predictions.csv
lb_std_predictions.csv
saved_models/
```

## Download Submission Files

To download the prediction files from Colab:

```python
from google.colab import files

files.download('lb_mu_predictions.csv')
files.download('lb_std_predictions.csv')
```

## Common Problems

### FileNotFoundError

This usually means the HDF5 or CSV files are not under `Hackathon_training/` in Drive.

Run this check:

```python
import os

for path in [
    'Hackathon_training/Training_SpectralData.hdf5',
    'Hackathon_training/Test_SpectralData.hdf5',
    'Hackathon_training/Training_targets.csv',
    'Hackathon_training/Training_supplementary_data.csv',
    'Hackathon_training/Training_supp_simulation_data.csv',
    'Hackathon_training/Test_supplementary_data.csv',
]:
    print(path, os.path.exists(path))
```

### GPU not being used

Check both:

1. Colab runtime is set to GPU.
2. You are running `competitive_solution_colab.py`, not the original CPU-only script.

### Out of memory or very slow training

Reduce these values further in the copied script:

```python
N_SPLITS = 2
PRETRAIN_EPOCHS = 5
FINETUNE_EPOCHS = 10
N_MC_SAMPLES = 5
```

### Notebook path issues

If the starter notebook cannot find files, run this first in a top cell:

```python
%cd /content/drive/MyDrive/ariel-inverse-modeling
```

## Suggested Order

1. Run the starter notebook once to confirm the dataset and baseline pipeline work.
2. Run the Colab copy of the competitive script.
3. Download `lb_mu_predictions.csv` and `lb_std_predictions.csv` for submission.