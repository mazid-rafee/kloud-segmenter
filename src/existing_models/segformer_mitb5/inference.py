import sys
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/mmsegmentation")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio as rio
import tacoreader
from mmengine.config import Config
from mmseg.models import build_segmentor


# -----------------------------
# Dataset (same as your training)
# -----------------------------
class CloudSegmentationDataset(Dataset):
    def __init__(self, taco_path, indices, selected_bands):
        self.dataset = tacoreader.load(taco_path)
        self.indices = indices
        self.selected_bands = selected_bands

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record = self.dataset.read(self.indices[idx])
        img_path = record.read(0)
        label_path = record.read(1)

        with rio.open(img_path) as src:
            img = src.read(indexes=self.selected_bands).astype(np.float32)

        img = torch.from_numpy(img / 3000.0).float()
        return img, 0   # dummy label (not needed)


# -----------------------------
# Inference speed function
# -----------------------------
def measure_inference_speed(model, loader, device, warmup=10, max_batches=40):
    model.eval()
    timings = []

    # Warm-up
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= warmup:
                break
            imgs = imgs.to(device)
            _ = model.encode_decode(imgs, [dict(img_shape=(512,512), ori_shape=(512,512))])

    # Timed passes
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= max_batches:
                break

            imgs = imgs.to(device)

            torch.cuda.synchronize()
            start = time.time()

            _ = model.encode_decode(imgs, [dict(img_shape=(512,512), ori_shape=(512,512))])

            torch.cuda.synchronize()
            end = time.time()

            timings.append(end - start)

    avg_batch_time = sum(timings) / len(timings)
    batch_size = next(iter(loader))[0].shape[0]

    latency = (avg_batch_time / batch_size) * 1000   # ms/image
    fps = batch_size / avg_batch_time

    return latency, fps


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
    indices = list(range(9000, 10000))   # test samples only
    selected_bands = list(range(1, 14))

    test_ds = CloudSegmentationDataset(taco_path, indices, selected_bands)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

    # Load model config
    cfg = Config(dict(
        model=dict(
            type='EncoderDecoder',
            backbone=dict(
                type='MixVisionTransformer',
                in_channels=13,
                embed_dims=64,
                num_stages=4,
                num_layers=[3,4,18,3],
                num_heads=[1,2,5,8],
                patch_sizes=[7,3,3,3],
                sr_ratios=[8,4,2,1],
                out_indices=(0,1,2,3),
                mlp_ratio=4,
                qkv_bias=True,
                norm_cfg=dict(type='LN', requires_grad=True)
            ),
            decode_head=dict(
                type='SegformerHead',
                in_channels=[64,128,320,512],
                in_index=[0,1,2,3],
                channels=512,
                num_classes=4,
                align_corners=False
            ),
            test_cfg=dict(mode='whole')
        )
    ))

    model = build_segmentor(cfg.model)
    model.init_weights()
    model = model.to(device)

    # Load weights (change if needed)
    ckpt_path = "src/results/checkpoints/segformer_mitb5_cloudsen12_l1c.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\n=== Measuring inference speed ===")
    latency, fps = measure_inference_speed(model, test_loader, device)

    print(f"\nLatency:   {latency:.2f} ms/image")
    print(f"Throughput (FPS): {fps:.2f}")
    print("\n=== DONE ===\n")
