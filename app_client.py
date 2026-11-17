"""app_client.py - web front-end (Streamlit) that talks to the LDCT-Recon inference SERVER.

Unlike app.py, this app does NOT load PyTorch or any model: all computation happens on the server, so it runs on any
computer that can reach the server (it only needs streamlit, requests, numpy).

    1) start the server (in one terminal):   uvicorn server:app --host 127.0.0.1 --port 8000
    2) start this app (in another):          streamlit run app_client.py

Server address / key can be preset with the environment variables LDCT_SERVER and LDCT_API_KEY.
Research prototype - not for clinical use.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import streamlit as st

from client import LDCTClient, LDCTError

st.set_page_config(page_title="LDCT-Recon client", page_icon="🩻", layout="wide")
st.title("LDCT-Recon - client")
st.caption("This page only uploads your slice and shows the answer. The model runs on the server.")

# ------------------------------------------------------------------------------------------- connection
with st.sidebar:
    st.header("Server")
    url = st.text_input("Address", os.environ.get("LDCT_SERVER", "http://127.0.0.1:8000"))
    key = st.text_input("API key (if the server needs one)", os.environ.get("LDCT_API_KEY", ""), type="password")
    client = LDCTClient(url, key or None, timeout=180)
    try:
        h = client.health()
        st.success(f"connected - {h['device']}" + (f" ({h['gpu']})" if h.get("gpu") else ""))
        st.caption(f"{h['models']} model(s) | {h['requests']} request(s) served | up {h['uptime_s']} s"
                   + (" | key required" if h["auth_required"] else " | OPEN server"))
    except LDCTError as exc:
        st.error(str(exc))
        st.stop()

try:
    models = client.models()
except LDCTError as exc:
    st.error(str(exc))
    st.stop()
if not models:
    st.warning("The server has no trained models yet. Train one with `python model.py train --organ liver`.")
    st.stop()

tab_run, tab_models, tab_about = st.tabs(["Enhance a scan", "Models on the server", "How this works"])

# ------------------------------------------------------------------------------------------- models
with tab_models:
    df = pd.DataFrame(models)[["organ", "arch", "label", "val_psnr", "baseline_psnr", "trained_epoch", "n_params",
                               "in_slices", "data_source", "default"]]
    st.dataframe(df, hide_index=True)
    st.caption("val_psnr / baseline_psnr: quick patch-level score on the validation patients (enhanced vs. raw low-dose); "
               "not the test result.")

# ------------------------------------------------------------------------------------------- enhance
def run_models(data, organ, archs, z, reference):
    """Ask the server once per chosen model (same slice) -> {arch: result}."""
    out = {}
    bar = st.progress(0.0, text="Waiting for the server...")
    for i, a in enumerate(archs):
        bar.progress(i / len(archs), text=f"Running {a} on the server ({i + 1}/{len(archs)})...")
        out[a] = client.enhance(data, organ, a, z=z, reference=reference)
    bar.empty()
    return out


with tab_run:
    c1, c2 = st.columns([1, 2])
    organs = sorted({m["organ"] for m in models})
    organ = c1.selectbox("Organ", organs)
    mine = [m for m in models if m["organ"] == organ]
    keys = [m["arch"] for m in mine]
    labels = {m["arch"]: m["label"] + (" (default)" if m["default"] else "") for m in mine}
    default_arch = next((m["arch"] for m in mine if m["default"]), keys[0])
    compare = c2.toggle("Compare several models on the same slice", value=False)
    if compare:
        archs = c2.multiselect("Models to run", keys, default=keys[: min(3, len(keys))], format_func=lambda a: labels[a])
    else:
        archs = [c2.selectbox("Model", keys, index=keys.index(default_arch), format_func=lambda a: labels[a])]
    up = st.file_uploader("Low-dose scan (.npy in HU, .dcm, .png)", type=["npy", "dcm", "png", "jpg", "jpeg"])
    ref_up = st.file_uploader("Optional: routine-dose reference of the same slice/volume (.npy) - adds metrics", type=["npy"])

    if up is not None:
        data: object = up.getvalue()
        n_slices, z = 1, None
        if up.name.lower().endswith(".npy"):
            arr = np.load(io.BytesIO(data), allow_pickle=False)
            data = arr
            n_slices = arr.shape[0] if arr.ndim == 3 else 1
            st.caption(f"array shape {arr.shape} - values are assumed to be Hounsfield units")
            if n_slices > 1:
                z = st.slider("Slice", 0, n_slices - 1, n_slices // 2)
        else:  # .dcm / .png: the client library takes a path, so use a temporary file
            import tempfile

            suffix = os.path.splitext(up.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
                fh.write(data)
            data = fh.name
        reference = np.load(io.BytesIO(ref_up.getvalue()), allow_pickle=False) if ref_up is not None else None

        if st.button("Enhance on the server", type="primary", disabled=not archs):
            try:
                st.session_state["results"] = run_models(data, organ, archs, z, reference)
            except LDCTError as exc:
                st.error(str(exc))
        if isinstance(data, str):
            try:
                os.unlink(data)
            except OSError:
                pass

    results = st.session_state.get("results")
    if results:
        first = next(iter(results.values()))
        m = first.meta
        st.caption(f"slice {m['slice']} of {m['n_slices']} | {m['shape'][0]}x{m['shape'][1]} px")
        for r in results.values():
            if r.meta.get("warning"):
                st.warning(f"{r.meta['model']['label']}: {r.meta['warning']}")
        panels = [("Low dose (input)", first.input_png)]
        panels += [(f"{r.meta['model']['label']}  ({r.meta['inference_ms']:.0f} ms)", r.png) for r in results.values()]
        if first.reference_png:
            panels.append(("Routine dose (reference)", first.reference_png))
        per_row = 4
        for i in range(0, len(panels), per_row):
            row = panels[i:i + per_row]
            for col, (title, png) in zip(st.columns(per_row), row):
                col.image(png, caption=title)

        if first.metrics:
            if len(results) == 1:
                k = first.metrics
                a_, b_, c_ = st.columns(3)
                a_.metric("PSNR", f"{k['psnr_out']:.2f} dB", f"{k['psnr_out'] - k['psnr_in']:+.2f}")
                b_.metric("SSIM", f"{k['ssim_out']:.4f}", f"{k['ssim_out'] - k['ssim_in']:+.4f}")
                c_.metric("RMSE", f"{k['rmse_out']:.1f} HU", f"{k['rmse_out'] - k['rmse_in']:+.1f}", delta_color="inverse")
            else:
                k0 = first.metrics
                rows = [dict(model="Low dose (input)", psnr_db=k0["psnr_in"], ssim=k0["ssim_in"], rmse_hu=k0["rmse_in"], server_ms=None)]
                rows += [dict(model=r.meta["model"]["label"], psnr_db=r.metrics["psnr_out"], ssim=r.metrics["ssim_out"],
                              rmse_hu=r.metrics["rmse_out"], server_ms=r.meta["inference_ms"]) for r in results.values()]
                st.dataframe(pd.DataFrame(rows).round(4), hide_index=True)
                st.caption("One slice only: a single-slice ranking is not evidence that one model is better. "
                           "The test-set comparison (model.py compare / cv.py) is the real result.")
        else:
            st.caption("Upload a routine-dose reference to see PSNR / SSIM / RMSE.")

        st.markdown("**Downloads**")
        for key, r in results.items():
            d1, d2, _ = st.columns([1, 1, 3])
            d1.download_button(f"{key} PNG", r.png, f"enhanced_{key}.png", "image/png", key=f"png_{key}")
            if r.enhanced is not None:
                buf = io.BytesIO()
                np.save(buf, r.enhanced)
                d2.download_button(f"{key} HU (.npy)", buf.getvalue(), f"enhanced_{key}.npy", key=f"npy_{key}")

# ------------------------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Client-server architecture**

* The **server** (`server.py`) holds PyTorch, the GPU and the trained checkpoints. It exposes a small REST API.
* This **client** sends the low-dose slice (only the slice you picked plus two neighbours on each side) and receives
  the enhanced slice, preview PNGs and, if you supplied a reference, PSNR / SSIM / RMSE.
* Because the model lives in one place, many users can share one GPU, and a user needs no PyTorch or GPU.

Research prototype - not for clinical use.""")
