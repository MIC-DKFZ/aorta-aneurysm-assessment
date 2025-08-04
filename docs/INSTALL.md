<!-- 
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
-->

# Install instructions

## Requirements

- Only linux is fully supported (tested on Ubuntu 20.04) [for prediction tasks]
- Python 3.9-3.11 (eitherwise use conda environment with python=3.9 or above) (python 3.12 currently has some issues installing the dependencies -- we recommend sticking with python 3.9).
- 16GB RAM
- [Prediction/training-only] Nvidia GPU w/ at least 6GB VRAM; CUDA already set up so that commands like `nvidia-smi` work.

## Partial install [no nnU-Net training/predictions] (needs <1 minute)

- Activate any virtual/conda environment (optional).
- Then just run
    ```
    pip install -r requirements.txt
    pip install -e -v .
    ```
- Command `aorta_aneurysm_v1` should now work.

## Full install [linux-only; if prediction tasks are needed] (needs <5 minutes)

- Activate any virtual/conda environment (optional).
- [Install appropriate pytorch](https://pytorch.org/get-started/locally/) (2.2.2 tested) -- depends on your system. Command `nvidia-smi` should work.
- Then just run 
    ```
    bash install.sh
    ```
- Command `aorta_aneurysm_v1` should now work.


#### A note on prediction tasks

nnU-Net will nnUNetTrainerRN1TR1 (which contains the extra augmentations will be installed under `deps/` using the `install.sh` script).
