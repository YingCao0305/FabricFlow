# FabricFlow

FabricFlow is organized as a two-phase pipeline:

1. Segmentation
   - `nnUNetv2` for 2D semantic segmentation
   - `Cellpose` for warp-fiber instance segmentation
2. Reconstruction
   - slice-wise clustering and label propagation for consistent fiber IDs

The repository keeps the upstream model sources in place:

- `nnUNet-master/`
- `cellpose-main/`

FabricFlow-specific orchestration, data preparation, runtime scripts, and reconstruction code are separated from those upstream projects.

The examples below assume the repository root is `./FabricFlow`. You can clone the repository into any directory, but all commands are written relative to that root.

## Repository Layout

```text
./FabricFlow/
├── README.md
├── requirements.txt
├── requirements_semantic.txt
├── requirements_instance.txt
├── requirements_reconstruction.txt
├── run_seg.sh
├── run_reconstruction.sh
├── segmentation/
│   ├── pipeline.py
│   └── prepare_instance_seg_dataset.py
├── reconstruction/
│   ├── pipeline.py
│   ├── reconstruction_workflow.py
│   ├── nnunet_prediction/
│   ├── weft_semantic_seg_mask/
│   ├── warp_instance_seg_mask/
│   ├── warp_instance_reconstruction_values/
│   └── segmentation_results/
├── nnUNet-master/
└── cellpose-main/
```

## Environment Setup

FabricFlow uses three isolated Python environments:

- semantic segmentation environment
- instance segmentation environment
- reconstruction environment

The bash entrypoints assume the following Conda environment names by default:

- `FabricFlow_semantic`
- `FabricFlow_instance`
- `FabricFlow_reconstruction`

The repository provides both:

- a single optional all-in-one install: `requirements.txt`
- official per-stage installs:
  - `requirements_semantic.txt`
  - `requirements_instance.txt`
  - `requirements_reconstruction.txt`

For the segmentation environments, the dependency choices follow the bundled upstream projects. If you need to adjust package versions, check the official `nnUNetv2` and `Cellpose` code and documentation first:

- `nnUNetv2`: <https://github.com/MIC-DKFZ/nnUNet>
- `Cellpose`: <https://github.com/MouseLand/cellpose>

### Official Three-Environment Setup

Example with virtual environments:

```bash
cd ./FabricFlow

python -m venv .venv_semantic
source .venv_semantic/bin/activate
pip install -r requirements_semantic.txt
deactivate

python -m venv .venv_instance
source .venv_instance/bin/activate
pip install -r requirements_instance.txt
deactivate

python -m venv .venv_reconstruction
source .venv_reconstruction/bin/activate
pip install -r requirements_reconstruction.txt
deactivate
```

If you prefer Conda, create three Conda environments and install the same requirement files with `pip install -r ...`.

### Optional Single-Environment Install

For quick inspection or CI only:

```bash
cd ./FabricFlow
python -m venv .venv_all
source .venv_all/bin/activate
pip install -r requirements.txt
```

The runtime scripts are still designed around separate environments. The all-in-one environment is only a convenience option.

## Dataset Preparation

Before running segmentation, place the initial input data in:

```text
./FabricFlow/nnUNet-master/DATASET/nnUNet_raw/Dataset666_FiberSegmentation/
├── imagesTr/        # semantic train + val images, *_0000.nii.gz
├── labelsTr/        # semantic train + val labels, *.nii.gz
├── finetune_image/  # adhesion cases for semantic prediction
└── test_image/      # test images
```

Naming convention:

- image: `case_0000.nii.gz`
- label: `case.nii.gz`

`dataset.json` must match the actual number of training pairs, and `splits_final.json` must define non-overlapping train and validation cases.

After semantic prediction on `finetune_image/`, manually correct the predictions and save the corrected labels to:

`./FabricFlow/nnUNet-master/DATASET/nnUNet_raw/Dataset666_FiberSegmentation/finetune_prediction_manual_refine`

## Segmentation Phase

### Stage Overview

1. Semantic preprocessing and training with `nnUNetv2`
2. Instance pretraining dataset preparation
3. `Cellpose` pretraining on warp masks
4. Semantic prediction on `finetune_image/`
5. Manual correction of the predicted adhesion regions
6. Instance finetuning dataset preparation
7. `Cellpose` finetuning
8. Semantic prediction on `test_image/`
9. Test dataset preparation for `Cellpose`
10. `Cellpose` inference on the test set
11. Export final warp instance masks to `reconstruction/warp_instance_seg_mask/`

### Manual Refinement Reminder

After `finetune-predict`, copy the corrected semantic masks into:

`./FabricFlow/nnUNet-master/DATASET/nnUNet_raw/Dataset666_FiberSegmentation/finetune_prediction_manual_refine`

Do not skip this step. `run_seg.sh full` will stop after semantic finetune prediction if no refined labels are found.

### Segmentation Entrypoints

Python stage runner:

```bash
python segmentation/pipeline.py check --semantic-python ./.venv_semantic/bin/python --instance-python ./.venv_instance/bin/python
python segmentation/pipeline.py pretrain --semantic-python ./.venv_semantic/bin/python --instance-python ./.venv_instance/bin/python
python segmentation/pipeline.py finetune-predict --semantic-python ./.venv_semantic/bin/python --instance-python ./.venv_instance/bin/python
python segmentation/pipeline.py finetune-prepare --semantic-python ./.venv_semantic/bin/python --instance-python ./.venv_instance/bin/python
python segmentation/pipeline.py finetune-train --semantic-python ./.venv_semantic/bin/python --instance-python ./.venv_instance/bin/python
python segmentation/pipeline.py test --semantic-python ./.venv_semantic/bin/python --instance-python ./.venv_instance/bin/python
```

Bash runner:

```bash
bash run_seg.sh check
bash run_seg.sh pretrain
bash run_seg.sh finetune-predict
bash run_seg.sh finetune-train
bash run_seg.sh test
bash run_seg.sh full
```

`run_seg.sh` is the recommended public entrypoint. It writes:

- `segmentation_logs/run_seg_status.txt`
- `segmentation_logs/run_seg_monitor.txt`
- `segmentation_logs/run_seg_<timestamp>.log`

### Environment Variables for Segmentation

If your environment names or locations differ, set explicit Python paths:

```bash
export SEMANTIC_PYTHON=./.venv_semantic/bin/python
export INSTANCE_PYTHON=./.venv_instance/bin/python
```

Common runtime overrides:

```bash
export NNUNET_TRAINER=nnUNetTrainer
export CELLPOSE_PRETRAIN_EPOCHS=500
export CELLPOSE_FINETUNE_EPOCHS=100
export CELLPOSE_DIAM_MEAN=25
```

## Reconstruction Phase

The reconstruction phase consumes the segmentation outputs written to:

- `./FabricFlow/reconstruction/warp_instance_seg_mask/`
- `./FabricFlow/reconstruction/weft_semantic_seg_mask/`
- `./FabricFlow/reconstruction/nnunet_prediction/`

Python entrypoint:

```bash
python reconstruction/pipeline.py --base-dir ./reconstruction
```

Bash entrypoint:

```bash
bash run_reconstruction.sh
```

Outputs:

- `./FabricFlow/reconstruction/warp_instance_reconstruction_values/`
- `./FabricFlow/reconstruction/segmentation_results/`

Logs:

- `reconstruction_logs/run_reconstruction_status.txt`
- `reconstruction_logs/run_reconstruction_monitor.txt`
- `reconstruction_logs/run_reconstruction_<timestamp>.log`

### Environment Variables for Reconstruction

```bash
export RECON_PYTHON=./.venv_reconstruction/bin/python
export ROTATION_ANGLE=95
```

## Third-Party Attribution

This repository integrates third-party model code and does not claim authorship of those implementations.

`nnUNetv2`

- upstream: <https://github.com/MIC-DKFZ/nnUNet>
- bundled here as `nnUNet-master/`
- upstream license: Apache 2.0
- bundled license file: `nnUNet-master/LICENSE`
- bundled notice file: `nnUNet-master/NOTICE.md`

`Cellpose`

- upstream: <https://github.com/MouseLand/cellpose>
- bundled here as `cellpose-main/`
- upstream license: BSD-style license
- bundled license file: `cellpose-main/LICENSE`
- bundled notice file: `cellpose-main/NOTICE.md`

Keep the original upstream licenses and citations when redistributing the repository.
