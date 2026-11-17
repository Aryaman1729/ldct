### liver: resunet, patient-level cross-validation (15 patients, each tested once by a model that never saw it)

| Metric | Low-dose input | Model output | Mean gain | Patients improved | Wilcoxon p (per patient) |
|---|---|---|---|---|---|
| PSNR (dB) | 30.23 ± 2.77 | 34.35 ± 2.34 | 4.13 | 15 / 15 | 6.1e-05 |
| SSIM | 0.8655 ± 0.0878 | 0.9203 ± 0.0596 | 0.0548 | 15 / 15 | 6.1e-05 |
| RMSE (HU) | 13.48 ± 6.26 | 8.19 ± 3.15 | 5.29 | 15 / 15 | 6.1e-05 |

± is the standard deviation ACROSS PATIENTS (not across slices).