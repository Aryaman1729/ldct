### Liver / Abdomen: architecture comparison (test set, mean ± std over slices)

| Model | Params (M) | PSNR (dB) | SSIM | RMSE (HU) | ms/slice | Train (min) |
|---|---|---|---|---|---|---|
| Low-dose input | - | 29.35 | 0.8739 | 14.5 | - | - |
| **Residual U-Net (ours)** | 4.15 | **33.56 ± 2.61** | 0.9212 | 8.8 | 24 | 8 |
| U-Net | 1.93 | 33.00 ± 2.56 | 0.9141 | 9.4 | 14 | 5 |
| DnCNN | 0.56 | 33.09 ± 2.64 | 0.9177 | 9.3 | 51 | 11 |
| RED-CNN | 1.85 | 33.27 ± 2.65 | 0.9189 | 9.1 | 127 | 18 |
| CNN (3-layer) | 0.07 | 32.41 ± 2.35 | 0.9144 | 9.9 | 7 | 2 |
