"""extras.py - classical image-processing baselines + "what is the model doing?" figures for LDCT-Recon.

Put this file in the SAME folder as dataset.py, model.py and app.py.  Run after training a model:

    python extras.py baselines --organ liver          # Gaussian / median / non-local means vs your model, on the test patients
    python extras.py explain   --organ liver          # 5 diagnostic figures
    python extras.py all       --organ liver          # both
Options: --arch resunet (default: the 'auto' choice)  --slices 10 (test slices per patient)

Outputs go to  results/<organ>/extras/ :
    baselines.md / baselines.json / baselines_visual.png
    1_residual.png 2_spectrum.png 3_features.png 4_receptive_field.png 5_edge_vs_flat.png
Needs: scikit-image (pip install scikit-image) for the non-local-means baseline; everything else is already installed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_erosion, gaussian_filter, gaussian_gradient_magnitude, median_filter

from dataset import ORGANS, PROJECT_HOME, cache_dir_for, get_splits, window_to_unit
from model import enhance_slice, load_model, pick_device, psnr, resolve_arch, rmse_hu, run_dir_for, ssim


# ------------------------------------------------------------------------------------------------ helpers
def tt(a) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a, np.float32))[None, None]


def test_patients(organ: str, per_patient: int, split: str = "test"):
    """Yield (patient key, low volume HU, full volume HU, chosen slice indices) for the patients of one split.
    Slices are spread evenly over the part of the scan that contains the body."""
    cache = cache_dir_for(organ)
    for key in get_splits(cache)[split]:
        d = cache / key
        full = np.load(d / "full.npy", mmap_mode="r")
        low = np.load(d / "low.npy").astype(np.float32)
        body = np.load(d / "body.npy")
        keep = [z for z in range(full.shape[0]) if body[z].mean() > 0.15]
        if keep:
            picks = np.linspace(0, len(keep) - 1, min(per_patient, len(keep))).astype(int)
            yield key, low, full, [keep[i] for i in picks]


def centre(low_vol: np.ndarray, z: int, k: int) -> np.ndarray:
    return low_vol[min(max(z, 0), low_vol.shape[0] - 1)]


def out_dir(organ: str) -> Path:
    d = PROJECT_HOME / "results" / organ / "extras"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_model(organ: str, arch: str):
    arch = resolve_arch(organ, arch)
    dev = pick_device()
    m, ck = load_model(run_dir_for(organ, arch) / "best.pt", dev)
    return m, ck, dev, arch


# ------------------------------------------------------------------------------------------------ baselines
def tune_filters(organ: str, lo: float, hi: float, n_slices: int = 4) -> dict:
    """Pick each classical filter's strength by best PSNR on the VALIDATION patients, so the baselines are
    as strong as they can be (a fair comparison) and the test patients are never used for tuning."""
    imgs = []
    for _, low, full, picks in test_patients(organ, n_slices, "val"):
        for z in picks[:n_slices]:
            imgs.append((window_to_unit(low[z], lo, hi), window_to_unit(np.asarray(full[z], np.float32), lo, hi)))
    score = lambda fn: float(np.mean([psnr(tt(fn(x)), tt(f)).item() for x, f in imgs]))  # noqa: E731
    best = {}
    best["gauss"] = max([0.5, 0.75, 1.0, 1.5, 2.0], key=lambda s: score(lambda x: gaussian_filter(x, s)))
    best["median"] = max([3, 5], key=lambda k: score(lambda x: median_filter(x, k)))
    try:
        from skimage.restoration import denoise_nl_means
        best["nlm"] = max([0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16], key=lambda h: score(
            lambda x: denoise_nl_means(x, h=h, sigma=0.0, patch_size=5, patch_distance=6, fast_mode=True).astype(np.float32)))
    except ImportError:
        pass
    print("filter settings tuned on validation patients:", best)
    return best


def run_baselines(organ: str, arch: str, per_patient: int) -> None:
    model, ck, dev, arch = get_model(organ, arch)
    lo, hi = ck["hu_window"]
    t = tune_filters(organ, lo, hi)
    methods = {"Low-dose input": lambda x: x, f"Gaussian (sigma={t['gauss']})": lambda x: gaussian_filter(x, t["gauss"]),
               f"Median ({t['median']}x{t['median']})": lambda x: median_filter(x, t["median"])}
    if "nlm" in t:
        from skimage.restoration import denoise_nl_means
        methods[f"Non-local means (h={t['nlm']})"] = lambda x: denoise_nl_means(
            x, h=t["nlm"], sigma=0.0, patch_size=5, patch_distance=6, fast_mode=True).astype(np.float32)
    else:
        print("scikit-image not installed -> non-local means skipped (pip install scikit-image)")
    names = list(methods) + [f"{arch} (deep model)"]
    rows = {n: [] for n in names}          # (patient, psnr, ssim, rmse_hu) per slice
    example = None
    for key, low, full, picks in test_patients(organ, per_patient):
        for z in picks:
            f = window_to_unit(np.asarray(full[z], np.float32), lo, hi)
            inp = window_to_unit(centre(low, z, ck["in_slices"]), lo, hi)
            outs = {n: fn(inp) for n, fn in methods.items()}
            outs[names[-1]] = window_to_unit(enhance_slice(model, ck, low, z, dev), lo, hi)
            for n, o in outs.items():
                rows[n].append((key, psnr(tt(o), tt(f)).item(), ssim(tt(o), tt(f)).item(), rmse_hu(tt(o), tt(f), (lo, hi)).item()))
            if example is None and len(rows[names[0]]) >= max(1, per_patient // 2):
                example = (key, z, outs, f)
    summary = {}
    for n, r in rows.items():
        a = np.array([x[1:] for x in r])
        per_pat = {}
        for x in r:
            per_pat.setdefault(x[0], []).append(x[1])
        pp = np.array([np.mean(v) for v in per_pat.values()])
        summary[n] = dict(psnr=a[:, 0].mean(), psnr_std=a[:, 0].std(), ssim=a[:, 1].mean(), rmse_hu=a[:, 2].mean(),
                          psnr_per_patient_mean=pp.mean(), psnr_per_patient_std=pp.std(), n_slices=len(a), n_patients=len(pp))
    d = out_dir(organ)
    (d / "baselines.json").write_text(json.dumps(summary, indent=2))
    lines = ["| Method | PSNR (dB) | SSIM | RMSE (HU) | PSNR per patient (mean ± std) |", "|---|---|---|---|---|"]
    for n, s in summary.items():
        lines.append(f"| {n} | {s['psnr']:.2f} ± {s['psnr_std']:.2f} | {s['ssim']:.4f} | {s['rmse_hu']:.1f} | "
                     f"{s['psnr_per_patient_mean']:.2f} ± {s['psnr_per_patient_std']:.2f} |")
    s0 = next(iter(summary.values()))
    txt = (f"### {ORGANS[organ].title}: classical filters vs deep model (test set: {s0['n_patients']} patients, "
           f"{s0['n_slices']} slices)\n\n" + "\n".join(lines) + "\n")
    (d / "baselines.md").write_text(txt)
    print(txt)
    key, z, outs, f = example
    fig, ax = plt.subplots(2, len(outs) + 1, figsize=(3 * (len(outs) + 1), 6.2))
    for j, (n, im) in enumerate(list(outs.items()) + [("Routine dose (target)", f)]):
        ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=1); ax[0, j].set_title(n, fontsize=9); ax[0, j].axis("off")
        ax[1, j].imshow(np.abs(im - f) * (hi - lo), cmap="magma", vmin=0, vmax=60); ax[1, j].set_title("|error| (HU, 0-60)", fontsize=8); ax[1, j].axis("off")
    fig.suptitle(f"test patient {key}, slice {z}", fontsize=10)
    fig.tight_layout(); fig.savefig(d / "baselines_visual.png", dpi=130); plt.close(fig)
    print("saved", d / "baselines_visual.png")


# ------------------------------------------------------------------------------------------------ explanation
def radial_spectrum(img: np.ndarray, nbins: int = 48):
    f = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    p = np.abs(f) ** 2 / img.size
    h, w = img.shape
    yy, xx = np.indices((h, w))
    r = np.hypot((yy - h / 2) / h, (xx - w / 2) / w)
    bins = np.linspace(0, 0.5, nbins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    prof = np.array([p.ravel()[idx == i].mean() if np.any(idx == i) else np.nan for i in range(nbins)])
    return 0.5 * (bins[:-1] + bins[1:]), prof


def run_explain(organ: str, arch: str, per_patient: int) -> None:
    model, ck, dev, arch = get_model(organ, arch)
    lo, hi = ck["hu_window"]; rng_hu = hi - lo; k = ck["in_slices"]
    d = out_dir(organ)
    data = list(test_patients(organ, per_patient))
    key, low, full, picks = data[0]
    z = picks[len(picks) // 2]
    f = window_to_unit(np.asarray(full[z], np.float32), lo, hi)
    inp = window_to_unit(centre(low, z, k), lo, hi)
    out = window_to_unit(enhance_slice(model, ck, low, z, dev), lo, hi)
    zs = np.clip(z + np.arange(k) - k // 2, 0, low.shape[0] - 1)
    xin = torch.from_numpy(window_to_unit(low[zs], lo, hi))[None].to(dev)

    # 1) what was removed
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for a, im, t, kw in [(ax[0], inp, "low-dose input", dict(cmap="gray", vmin=0, vmax=1)), (ax[1], out, "model output", dict(cmap="gray", vmin=0, vmax=1)),
                         (ax[2], (inp - out) * rng_hu, "what the model removed (HU)", dict(cmap="coolwarm", vmin=-80, vmax=80)),
                         (ax[3], (inp - f) * rng_hu, "true noise: input - routine dose (HU)", dict(cmap="coolwarm", vmin=-80, vmax=80))]:
        a.imshow(im, **kw); a.set_title(t, fontsize=9); a.axis("off")
    fig.tight_layout(); fig.savefig(d / "1_residual.png", dpi=130); plt.close(fig)

    # 2) noise power spectrum (average over the chosen test slices)
    P = {"input noise": [], "output error": []}
    for kk, lw, fl, pk in data:
        for zz in pk[:: max(1, len(pk) // 3)]:
            ff = window_to_unit(np.asarray(fl[zz], np.float32), lo, hi)
            ii = window_to_unit(centre(lw, zz, k), lo, hi)
            oo = window_to_unit(enhance_slice(model, ck, lw, zz, dev), lo, hi)
            fr, p1 = radial_spectrum((ii - ff) * rng_hu); _, p2 = radial_spectrum((oo - ff) * rng_hu)
            P["input noise"].append(p1); P["output error"].append(p2)
    plt.figure(figsize=(5.5, 4))
    for n, v in P.items():
        plt.semilogy(fr, np.nanmean(v, 0), label=n)
    plt.xlabel("spatial frequency (cycles/pixel)"); plt.ylabel("power (HU^2 px^2)"); plt.title("Noise power spectrum")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(d / "2_spectrum.png", dpi=130); plt.close()

    # 3) feature maps (works for networks with enc / mid / dec stages, e.g. the U-Nets and Restormer-lite)
    stages = {}
    if hasattr(model, "enc") and hasattr(model, "dec") and isinstance(model.enc, torch.nn.ModuleList):
        stages.update({f"enc{i}": m for i, m in enumerate(model.enc)})
        if hasattr(model, "mid"):
            stages["mid"] = model.mid
        stages.update({f"dec{i}": m for i, m in enumerate(model.dec)})
    if stages:
        feats, hs = {}, []
        for n, m in stages.items():
            hs.append(m.register_forward_hook(lambda mod, i, o, n=n: feats.__setitem__(n, o.detach()[0].float().cpu().numpy())))
        with torch.no_grad():
            model(xin)
        for h in hs:
            h.remove()
        fig, ax = plt.subplots(2, len(feats), figsize=(2.6 * len(feats), 5.4))
        for j, (n, fm) in enumerate(feats.items()):
            ax[0, j].imshow(np.abs(fm).mean(0), cmap="viridis"); ax[0, j].set_title(f"{n} ({fm.shape[0]}ch, {fm.shape[1]}px)", fontsize=8); ax[0, j].axis("off")
            top = int(np.argmax(np.abs(fm).mean((1, 2)))); ax[1, j].imshow(fm[top], cmap="gray"); ax[1, j].set_title(f"strongest channel #{top}", fontsize=8); ax[1, j].axis("off")
        fig.tight_layout(); fig.savefig(d / "3_features.png", dpi=130); plt.close(fig)
    else:
        print(f"(feature maps skipped: {arch} has no enc/mid/dec stages)")

    # 4) effective receptive field of the centre pixel
    xr = xin.clone().requires_grad_(True)
    model.zero_grad(); y = model(xr); H, W = y.shape[-2:]
    y[0, 0, H // 2, W // 2].backward()
    g = xr.grad[0].abs().amax(0).cpu().numpy(); g /= g.max() + 1e-12
    ys, xs = np.nonzero(g > 0.01); extent = int(max(np.ptp(ys), np.ptp(xs)) + 1) if len(ys) else 0
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(g ** 0.5, cmap="inferno"); ax[0].set_title(f"influence on centre pixel (>1% extent ~ {extent}px)", fontsize=9); ax[0].axis("off")
    ax[1].semilogy(np.arange(W) - W // 2, g[H // 2] + 1e-6); ax[1].set_title("profile through centre row"); ax[1].set_xlabel("pixels from centre"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(d / "4_receptive_field.png", dpi=130); plt.close(fig)

    # 5) error in flat vs edge regions
    edge = gaussian_gradient_magnitude(f, 1.0); body = f > 0.02
    edge_m = (edge > np.percentile(edge[body], 85)) & body
    flat_m = binary_erosion((edge < np.percentile(edge[body], 40)) & body, iterations=2)
    stats = {nm: (float(np.sqrt((((inp - f) * rng_hu)[m] ** 2).mean())), float(np.sqrt((((out - f) * rng_hu)[m] ** 2).mean())))
             for nm, m in [("flat regions", flat_m), ("edge regions", edge_m)]}
    plt.figure(figsize=(5, 4)); xs_ = np.arange(2)
    plt.bar(xs_ - .18, [v[0] for v in stats.values()], .36, label="input error (RMS HU)")
    plt.bar(xs_ + .18, [v[1] for v in stats.values()], .36, label="output error (RMS HU)")
    plt.xticks(xs_, list(stats)); plt.legend(); plt.title("Error in flat vs edge regions"); plt.tight_layout()
    plt.savefig(d / "5_edge_vs_flat.png", dpi=130); plt.close()
    print(f"receptive-field extent (>1% influence): ~{extent} px")
    print({n: (round(v[0], 1), round(v[1], 1)) for n, v in stats.items()}, "(input error, output error) in HU")
    print("figures ->", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["baselines", "explain", "all"])
    ap.add_argument("--organ", required=True, choices=list(ORGANS))
    ap.add_argument("--arch", default="auto")
    ap.add_argument("--slices", type=int, default=10, help="test slices per patient")
    a = ap.parse_args()
    if a.command in ("baselines", "all"):
        run_baselines(a.organ, a.arch, a.slices)
    if a.command in ("explain", "all"):
        run_explain(a.organ, a.arch, a.slices)