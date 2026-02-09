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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on:", device)


# ----------------------------------------------------------
# SAME DATASET CLASS YOU USED IN TRAINING/EVALUATION
# ----------------------------------------------------------
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

        # Only image is needed for inference
        img = torch.from_numpy(img / 3000.0).float()
        return img


# ----------------------------------------------------------
# EXACT SegFormer MIT-B5 CONFIG (unchanged)
# ----------------------------------------------------------
cfg = Config(dict(
    model=dict(
        type='EncoderDecoder',
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=13,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 4, 18, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            norm_cfg=dict(type='LN', requires_grad=True),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'
            )
        ),
        decode_head=dict(
            type='SegformerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=512,
            dropout_ratio=0.1,
            num_classes=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False
        ),
        test_cfg=dict(mode='whole')
    )
))


# ----------------------------------------------------------
# FUNCTION TO MEASURE INFERENCE SPEED
# ----------------------------------------------------------
def measure_inference_speed(model, loader, device, warmup=10, max_batches=40):
    model.eval()
    timings = []

    # ---- Warm-up ----
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if i >= warmup:
                break
            imgs = imgs.to(device)
            _ = model.encode_decode(imgs, [dict(img_shape=(512, 512),
                                               ori_shape=(512, 512))])

    # ---- Timed inference ----
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if i >= max_batches:
                break

            imgs = imgs.to(device)
            torch.cuda.synchronize()
            start = time.time()

            _ = model.encode_decode(imgs, [dict(img_shape=(512, 512),
                                               ori_shape=(512, 512))])

            torch.cuda.synchronize()
            end = time.time()
            timings.append(end - start)

    avg_batch_time = sum(timings) / len(timings)
    batch_size = next(iter(loader)).shape[0]

    latency_ms = (avg_batch_time / batch_size) * 1000
    fps = batch_size / avg_batch_time

    return latency_ms, fps


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    # Load test set exactly the same way as your training script
    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
    test_indices = list(range(9000, 10000))
    selected_bands = list(range(1, 14))

    test_ds = CloudSegmentationDataset(taco_path, test_indices, selected_bands)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

    # Build model
    model = build_segmentor(cfg.model)
    model.init_weights()
    model = model.to(device)

    # Load your trained weights (BEST MODEL)
    ckpt = "results/best_model_segformer_mitb5_cloudsen12_l1c.pth"
    print("Loading:", ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    print("\n=== Measuring inference speed ===")
    latency, fps = measure_inference_speed(model, test_loader, device)

    print(f"\nLatency: {latency:.2f} ms/image")
    print(f"Throughput: {fps:.2f} FPS")
    print("\nDONE\n")
