"""app.py - Streamlit demo for LDCT-Recon (low-dose CT enhancement, chest + liver).

    streamlit run app.py

Tabs: 1 Dataset (fetch from TCIA) | 2 Train (per organ, per architecture, GPU) | 3 Compare architectures |
      4 Enhance a scan | About.   Training runs in a background process (model.py train),
so the page stays responsive and you can watch the curves live.  Research prototype - not for clinical use.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from dataset import ORGANS, PROJECT_HOME, cache_dir_for, get_splits, load_index
from model import ARCHS, enhance_slice, load_model, psnr, resolve_arch, rmse_hu, run_dir_for, ssim

SRC = Path(__file__).resolve().parent      # code (scripts are launched from here)
ROOT = PROJECT_HOME                       # data/, runs/, results/
st.set_page_config(page_title="LDCT-Recon", page_icon="🩻", layout="wide")
st.markdown("""
<style>
 .block-container {padding-top: 1.5rem; max-width: 1250px;}
 .hero {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem; color: #fff;
        background: linear-gradient(120deg, #0f2a43 0%, #1d5b79 100%);}
 .hero h1 {margin: 0; font-size: 1.8rem; color: #fff;} .hero p {margin: .25rem 0 0; opacity: .85;}
 .pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; background:#e8f1f8; color:#164a68; margin-right:6px;}
</style>
<div class="hero"><h1>LDCT-Recon</h1>
<p>Low-dose CT enhancement &middot; chest and liver &middot; six architectures compared &middot; GPU training</p></div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- helpers
def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def read_history(rd: Path) -> pd.DataFrame:
    p = rd / "history.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return pd.DataFrame(rows)


def launch(cmd: list[str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=SRC)


def alive(p) -> bool:
    return p is not None and p.poll() is None


def tail(path: Path, n=14) -> str:
    try:
        return "\n".join(Path(path).read_text().splitlines()[-n:])
    except Exception:
        return ""


def trained_archs(organ: str) -> list[str]:
    return [a for a in ARCHS if (run_dir_for(organ, a) / "best.pt").exists()]


@st.cache_resource(show_spinner=False)
def get_model(path: str, mtime: float, device: str):
    return load_model(path, torch.device(device))


if "procs" not in st.session_state:
    st.session_state.procs = {}
P = st.session_state.procs
CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"

# ------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    organ = st.radio("Organ", list(ORGANS), format_func=lambda k: ORGANS[k].title)
    cfg = ORGANS[organ]
    st.caption(cfg.description)
    st.markdown(f"<span class='pill'>low dose = {int(cfg.dose_fraction * 100)} % of routine</span>"
                f"<span class='pill'>window {cfg.hu_window[0]:.0f}..{cfg.hu_window[1]:.0f} HU</span>", unsafe_allow_html=True)
    st.divider()
    if CUDA:
        pr = torch.cuda.get_device_properties(0)
        st.success(f"GPU: {pr.name} ({pr.total_memory / 1e9:.0f} GB)")
    else:
        st.error("No CUDA GPU detected - training would run on CPU. See README (GPU setup).")
    for k, c in ORGANS.items():
        e = load_index(cache_dir_for(k))
        st.write(f"**{c.title}**: {len(e)} cases cached, {len(trained_archs(k))} models trained")
    st.caption("Research prototype. Not a medical device; not validated for diagnosis.")

tab_data, tab_train, tab_cmp, tab_infer, tab_about = st.tabs(
    ["1 · Dataset", "2 · Train", "3 · Compare architectures", "4 · Enhance a scan", "About"])

# ------------------------------------------------------------------------------------------- 1 dataset
with tab_data:
    st.subheader(f"{cfg.title} data - TCIA LDCT-and-Projection-data")
    entries = load_index(cache_dir_for(organ))
    c1, c2 = st.columns([3, 2])
    with c1:
        if entries:
            df = pd.DataFrame(entries)
            cols = [c for c in ["key", "source", "vendor", "kernel_full", "thickness_mm", "n_slices", "kvp"] if c in df]
            st.dataframe(df[cols], hide_index=True, height=300)
            st.success(f"{len(df)} cases from {df['patient_id'].nunique()} patients, {int(df['n_slices'].sum())} slice pairs "
                       f"(source: {', '.join(sorted(df['source'].unique()))}).")
            if "synthetic" in set(df["source"]):
                st.warning("Contains SYNTHETIC test data - delete data/cache before real experiments.")
        else:
            st.info("No data cached yet. Use the buttons on the right to fetch it from TCIA.")
    with c2:
        st.markdown("**Fetch through the TCIA API**")
        maxp = st.number_input("Max patients (0 = complete dataset)", 0, 500, 0, key="maxp")
        busy = alive(P.get("data"))
        if st.button("1) List what TCIA offers (no download)", disabled=busy):
            P["data"] = launch([sys.executable, "dataset.py", "list", "--organ", organ], ROOT / "data" / "logs" / "list.log")
            st.rerun()
        if st.button("2) Download + convert", type="primary", disabled=busy):
            cmd = [sys.executable, "dataset.py", "download", "--organ", organ, "--yes"]
            if maxp:
                cmd += ["--max-patients", str(int(maxp))]
            P["data"] = launch(cmd, ROOT / "data" / "logs" / "download.log")
            st.rerun()
        st.caption("Only the Siemens cases have low-dose images, so about 50 paired patients per organ are used. "
                   "Raw DICOM is deleted after conversion; re-running resumes.")
        folder = st.text_input("or: already-downloaded DICOM folder")
        if st.button("Convert folder", disabled=busy or not folder):
            P["data"] = launch([sys.executable, "dataset.py", "from-folder", "--organ", organ, "--dicom-dir", folder],
                               ROOT / "data" / "logs" / "folder.log")
            st.rerun()

    @st.fragment(run_every=3)
    def data_status():
        p = P.get("data")
        if p is None:
            return
        newest = max((ROOT / "data" / "logs").glob("*.log"), key=lambda f: f.stat().st_mtime, default=None)
        (st.info("Working...") if alive(p) else (st.success("Finished.") if p.returncode == 0 else st.error("Failed - see log.")))
        if newest:
            st.code(tail(newest, 18) or "(no output yet)", language="text")
    data_status()

    with st.expander("Which metadata is used?"):
        st.markdown("""
**Used for curation (not as network input):** series description (full vs low dose), slice position (pairing),
rescale slope/intercept (HU), thickness, kernel, vendor - saved per case in `meta.json`. Split is **by patient**.
**Not used:** age, diagnosis, lesion annotations (they are not available at inference and invite shortcut learning);
use them afterwards to evaluate lesion visibility.""")

# ------------------------------------------------------------------------------------------- 2 train
with tab_train:
    st.subheader(f"Train the {cfg.title} model")
    entries = load_index(cache_dir_for(organ))
    if not entries:
        st.warning("Fetch data first (tab 1).")
    a, b, c = st.columns(3)
    arch_opts = ["auto"] + list(ARCHS)
    arch_sel = a.selectbox("Architecture", arch_opts, format_func=lambda k: "auto (best from comparison)" if k == "auto" else ARCHS[k]["label"])
    epochs = a.number_input("Epochs", 1, 1000, cfg.epochs)
    batch = b.number_input("Batch size", 1, 512, cfg.batch_size)
    patch = b.selectbox("Patch size", [64, 96, 128, 192, 256], index=[64, 96, 128, 192, 256].index(cfg.patch_size))
    lr = c.number_input("Learning rate", 1e-6, 1e-2, cfg.lr, format="%.6f")
    workers = c.number_input("Data-loader workers", 0, 32, 4)
    amp = st.checkbox("Mixed precision (faster on GPU)", True)
    arch = resolve_arch(organ, arch_sel)
    rd = run_dir_for(organ, arch)
    key = f"train_{organ}_{arch}"
    running = alive(P.get(key))
    b1, b2, b3 = st.columns([1, 1, 1])
    if b1.button("Start training", type="primary", disabled=running or not entries):
        cmd = [sys.executable, "model.py", "train", "--organ", organ, "--arch", arch, "--epochs", str(int(epochs)), "--batch-size", str(int(batch)),
               "--patch-size", str(patch), "--lr", str(lr), "--num-workers", str(int(workers))]
        if not amp:
            cmd.append("--no-amp")
        P[key] = launch(cmd, rd / "train.log")
        st.rerun()
    if b2.button("Resume", disabled=running or not (rd / "last.pt").exists()):
        P[key] = launch([sys.executable, "model.py", "train", "--organ", organ, "--arch", arch, "--resume", "--epochs", str(int(epochs)),
                         "--num-workers", str(int(workers))], rd / "train.log")
        st.rerun()
    if b3.button("Stop", disabled=not running):
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "STOP").write_text("stop")
        st.toast("Stop requested - finishing the current epoch.")
    st.caption(f"Training **{ARCHS[arch]['label']}** → `{rd.relative_to(ROOT)}`")

    @st.fragment(run_every=3)
    def train_status():
        status = read_json(rd / "status.json", {})
        hist = read_history(rd)
        state = status.get("state")
        if state == "failed":
            st.error(status.get("error", "Training failed"))
        elif alive(P.get(key)) or state == "running":
            tot = status.get("total_epochs", 1)
            st.progress(min(1.0, status.get("epoch", 0) / max(1, tot)), text=f"Epoch {status.get('epoch', 0)} / {tot} on {status.get('device', '?')}")
        elif state in ("finished", "stopped"):
            st.success(f"Training {state}. Best validation PSNR {status.get('best_psnr', 0):.2f} dB.")
        if status.get("data_source") == "synthetic":
            st.warning("This run used SYNTHETIC data - pipeline test only.")
        if not hist.empty:
            base = status.get("baseline", {}).get("psnr")
            last = hist.iloc[-1]
            m = st.columns(4)
            m[0].metric("Val PSNR", f"{last['val_psnr']:.2f} dB", None if base is None else f"{last['val_psnr'] - base:+.2f} vs input")
            m[1].metric("Val SSIM", f"{last['val_ssim']:.4f}")
            m[2].metric("Val RMSE", f"{last['val_rmse_hu']:.1f} HU")
            m[3].metric("Parameters", f"{status.get('n_params', 0) / 1e6:.2f} M")
            g1, g2 = st.columns(2)
            g1.caption("Training loss")
            g1.line_chart(hist.set_index("epoch")[["loss", "l1"]])
            g2.caption("Validation PSNR (dB)")
            ch = hist.set_index("epoch")[["val_psnr"]].rename(columns={"val_psnr": "model"})
            if base is not None:
                ch["low-dose input"] = base
            g2.line_chart(ch)
        tm = status.get("test")
        if tm:
            st.markdown("**Held-out test patients**")
            t = st.columns(3)
            t[0].metric("PSNR", f"{tm['psnr_out']:.2f} dB", f"{tm['psnr_out'] - tm['psnr_in']:+.2f}")
            t[1].metric("SSIM", f"{tm['ssim_out']:.4f}", f"{tm['ssim_out'] - tm['ssim_in']:+.4f}")
            t[2].metric("RMSE", f"{tm['rmse_out']:.1f} HU", f"{tm['rmse_out'] - tm['rmse_in']:+.1f}", delta_color="inverse")
            if (rd / "test_example.png").exists():
                st.image(str(rd / "test_example.png"))
        lg = tail(rd / "train.log", 8)
        if lg:
            with st.expander("Log"):
                st.code(lg, language="text")
    train_status()

# ------------------------------------------------------------------------------------------- 3 compare
with tab_cmp:
    st.subheader(f"{cfg.title}: which architecture is best?")
    st.caption("Trains every selected architecture under identical conditions and evaluates on the same held-out patients. "
               "The winner (by validation PSNR) becomes the 'auto' choice for training and enhancement.")
    sel = st.multiselect("Architectures", list(ARCHS), default=list(ARCHS), format_func=lambda k: ARCHS[k]["label"])
    ep = st.number_input("Epochs per architecture", 1, 500, 40)
    busy = alive(P.get(f"cmp_{organ}"))
    if st.button("Run comparison", type="primary", disabled=busy or not sel or not load_index(cache_dir_for(organ))):
        P[f"cmp_{organ}"] = launch([sys.executable, "model.py", "compare", "--organ", organ, "--epochs", str(int(ep)),
                                    "--archs", *sel, "--reuse"], ROOT / "results" / organ / "compare.log")
        st.rerun()
    res = ROOT / "results" / organ

    @st.fragment(run_every=5)
    def cmp_status():
        if alive(P.get(f"cmp_{organ}")):
            st.info("Comparison running - see log below.")
        if (res / "compare.log").exists():
            with st.expander("Log", expanded=alive(P.get(f"cmp_{organ}"))):
                st.code(tail(res / "compare.log", 14), language="text")
        rows = read_json(res / "comparison.json")
        if rows:
            df = pd.DataFrame(rows)
            best = read_json(res / "best_arch.json", {})
            st.success(f"Best architecture: **{best.get('label', '?')}** (selected on validation PSNR)")
            show = df[["label", "params_m", "psnr_out", "ssim_out", "rmse_out", "infer_ms", "train_min"]].rename(columns={
                "label": "Model", "params_m": "Params (M)", "psnr_out": "PSNR (dB)", "ssim_out": "SSIM", "rmse_out": "RMSE (HU)",
                "infer_ms": "ms / slice", "train_min": "Train (min)"})
            st.dataframe(show.style.format({"Params (M)": "{:.2f}", "PSNR (dB)": "{:.2f}", "SSIM": "{:.4f}", "RMSE (HU)": "{:.1f}",
                                            "ms / slice": "{:.0f}", "Train (min)": "{:.0f}"}), hide_index=True)
            for f, cap in [("comparison_bars.png", "Metrics"), ("comparison_visual.png", "Same slice, every model")]:
                if (res / f).exists():
                    st.image(str(res / f), caption=cap)
            sig = read_json(res / "significance.json")
            if sig and sig.get("vs"):
                st.caption("Paired Wilcoxon test on per-slice PSNR (best vs others): " + "; ".join(
                    f"{ARCHS[k]['label']}: {v['mean_psnr_gain_db']:+.2f} dB, p={v['p_value']:.2g}" for k, v in sig["vs"].items()))
        else:
            st.info("No comparison results yet.")
    cmp_status()

# ------------------------------------------------------------------------------------------- 4 enhance
with tab_infer:
    st.subheader(f"Enhance a {cfg.title.lower()} low-dose scan")
    have = trained_archs(organ)
    if not have:
        st.info("No trained model for this organ yet (tab 2 or 3).")
    else:
        best = read_json(ROOT / "results" / organ / "best_arch.json", {}).get("arch")
        arch_i = st.selectbox("Model", have, index=have.index(best) if best in have else 0, format_func=lambda k: ARCHS[k]["label"])
        bp = run_dir_for(organ, arch_i) / "best.pt"
        model, ck = get_model(str(bp), bp.stat().st_mtime, DEVICE)
        lo, hi = ck["hu_window"]
        if ck["data_source"] == "synthetic":
            st.warning("This model was trained on synthetic data - interface test only.")
        src = st.radio("Input", ["Held-out test case", "Upload (DICOM / .npy / PNG)"], horizontal=True)
        low_vol = ref_vol = None
        z = 0
        if src == "Held-out test case":
            keys = get_splits(cache_dir_for(organ))["test"] if load_index(cache_dir_for(organ)) else []
            if keys:
                k = st.selectbox("Test case", keys)
                d = cache_dir_for(organ) / k
                low_vol = np.load(d / "low.npy").astype(np.float32)
                ref_vol = np.load(d / "full.npy", mmap_mode="r")
                z = st.slider("Slice", 0, low_vol.shape[0] - 1, low_vol.shape[0] // 2)
        else:
            up = st.file_uploader("Low-dose slice", type=["dcm", "npy", "png", "jpg", "jpeg"])
            if up is not None:
                nm = up.name.lower()
                if nm.endswith(".dcm"):
                    import pydicom

                    ds = pydicom.dcmread(io.BytesIO(up.getvalue()), force=True)
                    hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
                    low_vol = np.clip(hu, -1024, 3071)[None]
                elif nm.endswith(".npy"):
                    arr = np.load(io.BytesIO(up.getvalue())).astype(np.float32)
                    low_vol = arr[None] if arr.ndim == 2 else arr
                    st.caption("Assuming .npy values are Hounsfield units.")
                    if low_vol.shape[0] > 1:
                        z = st.slider("Slice", 0, low_vol.shape[0] - 1, low_vol.shape[0] // 2)
                else:
                    from PIL import Image

                    g = np.asarray(Image.open(io.BytesIO(up.getvalue())).convert("L"), np.float32) / 255.0
                    low_vol = (lo + g * (hi - lo))[None]
                    st.caption("PNG/JPG has no HU values: grey levels are mapped linearly onto this organ's window.")
        if low_vol is not None:
            with st.spinner("Enhancing..."):
                out_hu = enhance_slice(model, ck, low_vol, z, torch.device(DEVICE))
            inp = low_vol[z]
            ref = np.asarray(ref_vol[z], np.float32) if ref_vol is not None else None
            w1, w2 = st.columns(2)
            vmin = w1.number_input("Display min (HU)", value=float(lo))
            vmax = w2.number_input("Display max (HU)", value=float(hi))
            panels = [("Low dose (input)", inp), (f"Enhanced ({ARCHS[arch_i]['label']})", out_hu)] + ([("Routine dose (reference)", ref)] if ref is not None else [])
            fig, axs = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.2))
            for ax, (t, im) in zip(np.atleast_1d(axs), panels):
                ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(t)
                ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            if ref is not None:
                tt = lambda a: torch.from_numpy(np.clip((np.asarray(a, np.float32) - lo) / (hi - lo), 0, 1))[None, None]  # noqa: E731
                pi, po = psnr(tt(inp), tt(ref)).item(), psnr(tt(out_hu), tt(ref)).item()
                si, so = ssim(tt(inp), tt(ref)).item(), ssim(tt(out_hu), tt(ref)).item()
                ri, ro = rmse_hu(tt(inp), tt(ref), (lo, hi)).item(), rmse_hu(tt(out_hu), tt(ref), (lo, hi)).item()
                m = st.columns(3)
                m[0].metric("PSNR", f"{po:.2f} dB", f"{po - pi:+.2f}")
                m[1].metric("SSIM", f"{so:.4f}", f"{so - si:+.4f}")
                m[2].metric("RMSE", f"{ro:.1f} HU", f"{ro - ri:+.1f}", delta_color="inverse")
            png = io.BytesIO()
            plt.imsave(png, np.clip((out_hu - vmin) / (vmax - vmin), 0, 1), cmap="gray", format="png")
            npy = io.BytesIO()
            np.save(npy, out_hu.astype(np.float32))
            d1, d2 = st.columns(2)
            d1.download_button("Download enhanced PNG", png.getvalue(), f"enhanced_{organ}.png", "image/png")
            d2.download_button("Download enhanced HU (.npy)", npy.getvalue(), f"enhanced_{organ}.npy")

# ------------------------------------------------------------------------------------------- about
with tab_about:
    st.subheader("How it works")
    st.markdown("""
**Task** - supervised image-to-image denoising: a low-dose slice (plus its two neighbours, 2.5D) goes in, the
routine-dose slice of the same patient is the target.

**Data** - TCIA *LDCT-and-Projection-data* (Mayo Clinic). Fetched through the NBIA REST API; Siemens cases with both
full- and low-dose reconstructions (chest 10 %, liver 25 % dose); patient-level train/val/test split.

**Models** - CNN, DnCNN, RED-CNN, U-Net, Residual U-Net, Restormer-lite; same loss (L1 + SSIM + edge), optimiser and
budget; the best on the validation set is used for each organ.

**Metrics** - PSNR, SSIM, RMSE (HU) versus the routine-dose scan on held-out patients. These measure similarity, not
diagnostic quality; lesion detectability needs a reader study.""")
    st.caption("Data: McCollough et al., Low Dose CT Image and Projection Data, TCIA, doi:10.7937/9npb-2637 (CC BY 4.0). "
               "Acknowledge NIBIB grants EB017095 and EB017185.")