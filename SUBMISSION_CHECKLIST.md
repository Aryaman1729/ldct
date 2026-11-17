# Submission checklist

No one can promise a grade; graders differ. This list covers what most evaluators of a project like this look for:
a clear problem, correct method, honest evidence, reproducibility, and a student who can explain it.

## 1. Before you hand in

- [ ] Open `app_site.py` and fill in `PROJECT` at the top: `course`, `guide` (professor's name), `institution`. Leave `""` to hide a field.
- [ ] Run `python test_serving.py`; it must end with `ALL CHECKS PASSED`.
- [ ] Run `streamlit run app_site.py` and click every section once. Check the Results numbers match your report.
- [ ] Start the server and test the Live demo with one test patient (for example `data\cache\liver\L006\low.npy`, with `full.npy` as reference).
- [ ] Take screenshots of every site section (backup in case the laptop or browser misbehaves on the day).
- [ ] Record a 2 minute screen video of the demo as a second backup.

## 2. What to include in the submission

- [ ] Code: `dataset.py`, `model.py`, `extras.py`, `cv.py`, `server.py`, `client.py`, `app.py`, `app_client.py`, `app_site.py`, `test_serving.py`, `requirements*.txt`
- [ ] `README_PROJECT.md` (rename to `README.md`)
- [ ] `results\liver\` (tables and figures) so the site shows results without retraining
- [ ] The trained model `runs\liver\resunet\best.pt` (a few MB) so the demo works
- [ ] Report (PDF) and slides
- [ ] Do NOT include: the `.venv` folder, the DICOM data, `data\cache`, `last.pt` files, or `__pycache__`

## 3. Report outline (typical expectations)

1. Abstract
2. Introduction: problem and motivation (dose versus noise)
3. Dataset: Mayo LDCT, what you used, split, licence
4. Method: pipeline, 2.5D input, five architectures, loss, training
5. Experiments: metrics, classical baselines tuned on validation, cross-validation
6. Results: the three tables and the figures, with honest reading
7. Discussion and limitations
8. Conclusion and future work
9. References (list is on the site, Reproduce page)

## 4. Strengthen it before the deadline (highest value first)

1. **A small ablation** (2D versus 2.5D, with and without the edge and SSIM terms). It lets you answer "why did you design it like this?" with evidence.
2. **Different random seeds** for the top two models, so you can say whether the gap between them is real.
3. **A lesion-focused check**, even a simple one: compare a region containing a visible lesion before and after.

## 5. Be ready to answer

- Why a patient-level split? Why not tune on the test set? How were the classical filters made fair?
- What do PSNR, SSIM and RMSE mean, and what do they miss?
- Why is the gain over classical filters smaller than the gain over the raw scan?
- What does the noise power spectrum show, and where does the model fail?
- Why only the abdomen? (Answer honestly: it was chosen as the scope so it could be studied in depth; other regions are future work.)
- What would you do next?
