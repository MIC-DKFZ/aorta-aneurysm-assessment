#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
mkdir -p $SCRIPTPATH/deps

## REQUIREMENTS
echo ">> INSTALLING PYTHON REQUIREMENTS"; pip install -r $SCRIPTPATH/requirements.txt


## NNUNET (was v2.3.1)
echo ">> INSTALLING NNUNET"; git clone https://github.com/MIC-DKFZ/nnUNet.git $SCRIPTPATH/deps/nnUNet; \
    cd $SCRIPTPATH/deps/nnUNet; \
    git checkout tags/v2.3.1 -b v2.3.1; \
    pip install -e .; \
    cd -
cp $SCRIPTPATH/deploy/nnUNet-docker/nnUNetTrainerRN1TR1-v2.3.1.py \
   $SCRIPTPATH/deps/nnUNet/nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerRN1TR1.py

## THIS PACKAGE
pip install -e $SCRIPTPATH

echo ""; echo ""; echo ">> INSTALLATION COMPLETE. You can now use command: aorta_aneurysm_v1"; echo ""