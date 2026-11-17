### Liver / Abdomen: classical filters vs deep model (test set: 2 patients, 20 slices)

| Method | PSNR (dB) | SSIM | RMSE (HU) | PSNR per patient (mean ± std) |
|---|---|---|---|---|
| Low-dose input | 29.10 ± 3.08 | 0.8724 | 14.9 | 29.10 ± 2.16 |
| Gaussian (sigma=0.5) | 29.94 ± 2.50 | 0.8899 | 13.3 | 29.94 ± 1.66 |
| Median (3x3) | 29.73 ± 1.81 | 0.8950 | 13.3 | 29.73 ± 1.12 |
| Non-local means (h=0.04) | 31.16 ± 3.59 | 0.9001 | 12.1 | 31.16 ± 2.44 |
| resunet (deep model) | 33.36 ± 2.54 | 0.9202 | 9.0 | 33.36 ± 1.82 |
