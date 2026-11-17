"""cv.py - patient-level k-fold cross-validation for LDCT-Recon.

Why: your first result rests on 2 test patients. Here every one of your patients is a test patient exactly once,
so the final numbers come from ALL of them (15 patients -> 15 test results), and the statistics are done per
patient (the correct unit, because slices of one patient are almost identical).

Put this file in the SAME folder as dataset.py / model.py / extras.py.  Run (one line at a time):

    python cv.py --organ liver --arch resunet --folds 5 --epochs 30     # about 40 min on your RTX 4060 (5 x ~8 min)
    python cv.py --organ liver --report-only                            # rebuild tables/figure from finished folds

It is safe to stop (Ctrl+C) and restart: finished folds are skipped.  Your normal splits.json is backed up and
restored afterwards, so nothing else in the project changes.

Outputs -> results/<organ>/cv/ :  cv_per_patient.csv  cv_summary.md  cv_summary.json  cv_per_patient.png
Runs    -> runs/<organ>/cv/<arch>_fold<i>/
"""
from __future__ import annotations

import os

# Cap maths-library threads BEFORE numpy/torch load. Each data-loader worker would otherwise start its own
# pile of OpenBLAS threads, which can exhaust memory on Windows ("OpenBLAS error: Memory allocation failed").
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import PROJECT_HOME, cache_dir_for, load_index
from model import TrainArgs, evaluate_run, train

METRICS = [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("rmse", "RMSE (HU)")]


def make_folds(cache: Path, k: int, seed: int = 42):
    """Return (patients -> keys, list of k folds, each a dict train/val/test of patient ids)."""
    by_patient: dict[str, list[str]] = defaultdict(list)
    for e in load_index(cache):
        by_patient[e["patient_id"]].append(e["key"])
    pids = sorted(by_patient)
    if len(pids) < k + 2:
        raise SystemExit(f"Only {len(pids)} patients cached; need at least {k + 2} for {k} folds.")
    random.Random(seed).shuffle(pids)
    parts = [list(map(str, a)) for a in np.array_split(np.array(pids, dtype=object), k)]
    n_val = 2 if len(pids) >= 8 else 1
    folds = []
    for i in range(k):
        test = parts[i]
        nxt = parts[(i + 1) % k]
        val = nxt[:n_val]
        train_p = [p for p in pids if p not in test and p not in val]
        folds.append(dict(train=train_p, val=val, test=test))
    return by_patient, folds


def write_splits(cache: Path, by_patient, fold) -> None:
    sp = {name: [key for p in pats for key in by_patient[p]] for name, pats in fold.items()}
    sp["patients"] = fold
    (cache / "splits.json").write_text(json.dumps(sp, indent=2))


def run_cv(organ: str, arch: str, k: int, epochs: int, seed: int, force: bool, test_slices: int,
           num_workers: int) -> None:
    cache = cache_dir_for(organ)
    by_patient, folds = make_folds(cache, k, seed)
    split_file = cache / "splits.json"
    backup = cache / "splits.json.cv_backup"
    if split_file.exists():
        shutil.copy2(split_file, backup)
    try:
        for i, fold in enumerate(folds):
            out = PROJECT_HOME / "runs" / organ / "cv" / f"{arch}_fold{i}"
            if (out / "test_metrics.json").exists() and not force:
                print(f"[fold {i + 1}/{k}] already done - skipping")
                continue
            print(f"\n[fold {i + 1}/{k}] train {len(fold['train'])} | val {fold['val']} | test {fold['test']}")
            write_splits(cache, by_patient, fold)
            args = TrainArgs(organ=organ, arch=arch, out_dir=str(out), epochs=epochs, skip_test=True, seed=seed,
                             num_workers=num_workers)
            status = train(args)
            if status.get("state") == "failed":
                raise SystemExit(f"fold {i} failed: {status.get('error')}")
            evaluate_run(out, cache, max_slices=test_slices, make_figure=False)
            (out / "fold_patients.json").write_text(json.dumps(fold, indent=2))
    finally:
        if backup.exists():
            shutil.copy2(backup, split_file)
            backup.unlink()
        print("\n(original splits.json restored)")


def collect(organ: str, arch: str):
    """Per-patient mean metrics over all finished folds."""
    cache = cache_dir_for(organ)
    key2pid = {e["key"]: e["patient_id"] for e in load_index(cache)}
    rows = []
    for d in sorted((PROJECT_HOME / "runs" / organ / "cv").glob(f"{arch}_fold*")):
        f = d / "test_metrics.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        per = defaultdict(list)
        for r in res["per_slice"]:
            per[key2pid.get(r["case"], r["case"])].append(r)
        for pid, rs in per.items():
            row = dict(patient=pid, fold=d.name.split("fold")[-1], n_slices=len(rs))
            for m, _ in METRICS:
                row[f"{m}_in"] = float(np.mean([r[f"{m}_in"] for r in rs]))
                row[f"{m}_out"] = float(np.mean([r[f"{m}_out"] for r in rs]))
            rows.append(row)
    return sorted(rows, key=lambda r: r["patient"])


def summarise(organ: str, arch: str) -> None:
    from scipy.stats import wilcoxon

    rows = collect(organ, arch)
    if not rows:
        raise SystemExit("No finished folds found. Run without --report-only first.")
    out = PROJECT_HOME / "results" / organ / "cv"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cv_per_patient.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    summ, lines = {}, []
    lines.append(f"### {organ}: {arch}, patient-level cross-validation ({n} patients, each tested once by a model that never saw it)\n")
    lines.append("| Metric | Low-dose input | Model output | Mean gain | Patients improved | Wilcoxon p (per patient) |")
    lines.append("|---|---|---|---|---|---|")
    for m, label in METRICS:
        a = np.array([r[f"{m}_in"] for r in rows])
        b = np.array([r[f"{m}_out"] for r in rows])
        gain = (b - a) if m != "rmse" else (a - b)  # positive = better
        p = float(wilcoxon(gain).pvalue) if n >= 6 and np.any(gain != 0) else float("nan")
        summ[m] = dict(input_mean=a.mean(), input_std=a.std(ddof=1), output_mean=b.mean(), output_std=b.std(ddof=1),
                       mean_gain=gain.mean(), improved=int((gain > 0).sum()), n=n, wilcoxon_p=p)
        fmt = "{:.2f}" if m != "ssim" else "{:.4f}"
        lines.append(f"| {label} | {fmt.format(a.mean())} ± {fmt.format(a.std(ddof=1))} | "
                     f"{fmt.format(b.mean())} ± {fmt.format(b.std(ddof=1))} | {fmt.format(gain.mean())} | "
                     f"{int((gain > 0).sum())} / {n} | {p:.1e} |")
    lines.append("\n± is the standard deviation ACROSS PATIENTS (not across slices).")
    (out / "cv_summary.md").write_text("\n".join(lines))
    (out / "cv_summary.json").write_text(json.dumps(summ, indent=2))
    print("\n".join(lines))

    # figure: input vs output RMSE and PSNR per patient
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (m, label) in zip(axes, [("rmse", "RMSE (HU)  lower = better"), ("psnr", "PSNR (dB)  higher = better")]):
        x = np.arange(n)
        a = np.array([r[f"{m}_in"] for r in rows])
        b = np.array([r[f"{m}_out"] for r in rows])
        for i in range(n):
            ax.plot([i, i], [a[i], b[i]], color="0.7", lw=1)
        ax.scatter(x, a, color="#c0504d", label="low-dose input", zorder=3)
        ax.scatter(x, b, color="#1f7a8c", label=f"{arch} output", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([r["patient"] for r in rows], rotation=60, fontsize=8)
        ax.set_title(label)
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle(f"Every patient tested by a model trained without them ({n} patients)")
    fig.tight_layout()
    fig.savefig(out / "cv_per_patient.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organ", default="liver")
    ap.add_argument("--arch", default="resunet")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-slices", type=int, default=40, help="test slices per patient")
    ap.add_argument("--num-workers", type=int, default=2, help="data-loader processes (lower = less memory)")
    ap.add_argument("--force", action="store_true", help="redo folds that already finished")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        run_cv(a.organ, a.arch, a.folds, a.epochs, a.seed, a.force, a.test_slices, a.num_workers)
    summarise(a.organ, a.arch)


if __name__ == "__main__":
    main()
