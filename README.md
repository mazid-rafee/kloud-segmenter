# Cloud Segmentation

This repository contains code and experiments for cloud segmentation research.

## Publication

This work has been accepted at CAI 2026. The preprint is available on arXiv:
https://arxiv.org/pdf/2510.10802

## Repository Structure

- `src/` - Core training, inference, models, data loaders, utilities, and visuals
- `scripts/` - Dataset download/compile utilities and helpers
- `env_requirements/` - Environment export files for reproducible installs
- `mmsegmentation/`, `dinov2/`, `InternImage/`, `DCNv4/`, `fast_kan/` - External frameworks

## Setup

Environment exports are stored in `env_requirements/`:
- `satimage-env.txt` - Satellite image processing stack
- `mmseg-env.txt` - MMSegmentation stack
- `dinov2-env.txt` - DINOv2 stack
- `internimage-env.txt` - InternImage/DCN stack

You can create a conda environment from an export file:

```
conda create -n cloudseg --file env_requirements/mmseg-env.txt
```

## Data

CloudSEN12+ (TACO format) download script:

```
python scripts/download-coludsen12-taco.py
```

Optional mini subset compiler:

```
python scripts/compile-coludsen12-mini-taco.py
```

L8-Biome download script:

```
python scripts/download-l8-biome-torchgeo.py
```

By default, the scripts place data under `satellite_data/CloudSen12+/TACOs/`
and `data/L8-Biome/`.

## Training (MSCloudCAM)

Main training entrypoint:

```
python src/main.py --dataset cloudsen12_l1c --epochs 100 --gpu 0
```

Supported datasets: `cloudsen12_l1c`, `cloudsen12_l2a`, `l8biome`.

## Inference Speed (MSCloudCAM)

Run the inference speed benchmark:

```
python src/inference.py --gpu 0
```

The script expects checkpoints under `src/results/checkpoints/` and will
evaluate CloudSEN12 L1C/L2A and L8-Biome.

## Baselines and Existing Models

Baseline implementations live in `src/existing_models/`:
- SegFormer (`segformer_mitb5/`)
- Mask2Former (`mask2former_swin_t/`, `mask2former_dinov2_vitb14_reg/`)
- UPerNet + InternImage (`upernet_internimage_b_dcnv4/`)
- DeepLabV3+ UNet (`deeplabv3_unet/`)

## Results and Outputs

Training logs and checkpoints are written under `src/results/`.

