"""server.py - inference SERVER for LDCT-Recon (the "server" half of the client-server architecture).

The server owns everything heavy: PyTorch, the GPU and the trained checkpoints.  Clients (client.py, app_client.py,
or anything that can send HTTP) only upload a low-dose slice and get the enhanced slice back.

Put this file in the SAME folder as dataset.py / model.py (it imports your existing code) and run:

    uvicorn server:app --host 127.0.0.1 --port 8000

Optional environment variables (PowerShell:  $env:NAME="value"):
    LDCT_API_KEY        if set, every request except /health must send the header  X-API-Key: <that value>
    LDCT_MAX_UPLOAD_MB  largest accepted upload (default 64)
    LDCT_DEVICE         force "cpu" or "cuda" (default: cuda if available)
    LDCT_HOME           project folder holding runs/ (same variable dataset.py already uses)

API (interactive documentation at http://127.0.0.1:8000/docs while the server runs):
    GET  /health          is the server alive?  (no key needed)
    GET  /models          which trained models exist, with their metadata
    POST /v1/enhance      upload a low-dose slice/volume -> enhanced slice (+ metrics if a reference is sent)

Research prototype - not for clinical use.  Do not expose it to the internet without HTTPS and authentication.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from dataset import ORGANS, window_to_unit
from model import ARCHS, enhance_slice, load_model, pick_device, psnr, resolve_arch, rmse_hu, run_dir_for, ssim

API_VERSION = "1.0"
MAX_UPLOAD = int(float(os.environ.get("LDCT_MAX_UPLOAD_MB", "64")) * 1024 * 1024)
API_KEY = os.environ.get("LDCT_API_KEY") or None
MIN_SIDE, MAX_SIDE = 64, 2048  # accepted image height / width in pixels

log = logging.getLogger("ldct.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ------------------------------------------------------------------------------------------- model registry
class Registry:
    """Finds trained checkpoints (runs/<organ>/<arch>/best.pt) and keeps loaded models in memory.

    A model is loaded the first time it is requested and reloaded automatically if best.pt changes on disk
    (so you can retrain while the server is running)."""

    def __init__(self, device: torch.device):
        self.device = device
        self._cache: dict[tuple[str, str], tuple[float, torch.nn.Module, dict]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def available() -> list[tuple[str, str]]:
        return [(o, a) for o in ORGANS for a in ARCHS if (run_dir_for(o, a) / "best.pt").exists()]

    def get(self, organ: str, arch: str):
        path = run_dir_for(organ, arch) / "best.pt"
        if not path.exists():
            raise KeyError((organ, arch))
        mtime = path.stat().st_mtime
        with self._lock:
            hit = self._cache.get((organ, arch))
            if hit and hit[0] == mtime:
                return hit[1], hit[2]
            log.info("loading %s/%s from %s", organ, arch, path)
            model, ck = load_model(path, self.device)
            self._cache[(organ, arch)] = (mtime, model, ck)
            return model, ck


def describe(organ: str, arch: str, ck: dict) -> dict:
    return dict(organ=organ, arch=arch, label=ARCHS[arch]["label"], n_params=ck.get("n_params"),
                in_slices=ck["in_slices"], hu_window=list(ck["hu_window"]), data_source=ck.get("data_source"),
                trained_epoch=ck.get("epoch"), val_psnr=ck.get("val_psnr"), baseline_psnr=ck.get("baseline_psnr"))


# ----------------------------------------------------------------------------------------------- app state
DEVICE = pick_device(os.environ.get("LDCT_DEVICE") or None)
REGISTRY = Registry(DEVICE)
INFER_LOCK = threading.Lock()  # one GPU job at a time: a laptop GPU is not made for parallel inference
STATS = dict(started=time.time(), requests=0, errors=0)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    found = Registry.available()
    log.info("device: %s | %d trained model(s) found: %s", DEVICE, len(found), found)
    for organ, arch in found:  # load now so the first request is fast
        try:
            REGISTRY.get(organ, arch)
        except Exception as exc:  # a broken checkpoint must not stop the server
            log.warning("could not preload %s/%s: %s", organ, arch, exc)
    if API_KEY is None:
        log.warning("LDCT_API_KEY is not set: the server is OPEN to anyone who can reach it.")
    yield


app = FastAPI(title="LDCT-Recon inference server", version=API_VERSION, lifespan=lifespan,
              description="Low-dose CT enhancement as a service. Research prototype - not for clinical use.")


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY is not None and not (x_api_key and secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="missing or wrong API key (send header X-API-Key)")


# ---------------------------------------------------------------------------------------------- helpers
def read_upload(up: UploadFile) -> bytes:
    raw = up.file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail=f"upload larger than {MAX_UPLOAD // 2**20} MB")
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    return raw


def to_volume(name: str, raw: bytes, window: tuple[float, float]) -> np.ndarray:
    """Decode an upload into an (N, H, W) float32 volume in Hounsfield units."""
    name = (name or "").lower()
    try:
        if name.endswith(".dcm"):
            import pydicom

            ds = pydicom.dcmread(io.BytesIO(raw), force=True)
            slope, icpt = float(getattr(ds, "RescaleSlope", 1)), float(getattr(ds, "RescaleIntercept", 0))
            vol = np.clip(ds.pixel_array.astype(np.float32) * slope + icpt, -1024, 3071)[None]
        elif name.endswith(".npy"):
            vol = np.load(io.BytesIO(raw), allow_pickle=False).astype(np.float32)  # allow_pickle=False: never run uploaded code
        elif name.endswith((".png", ".jpg", ".jpeg")):
            from PIL import Image

            g = np.asarray(Image.open(io.BytesIO(raw)).convert("L"), np.float32) / 255.0
            vol = (window[0] + g * (window[1] - window[0]))[None]  # grey levels -> this organ's HU window
        else:
            raise HTTPException(status_code=400, detail="file must be .dcm, .npy, .png or .jpg")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not read '{name}': {type(exc).__name__}: {exc}") from None
    if vol.ndim == 2:
        vol = vol[None]
    if vol.ndim != 3:
        raise HTTPException(status_code=400, detail=f"expected a 2-D slice or a 3-D (slices, H, W) volume, got shape {vol.shape}")
    n, h, w = vol.shape
    if not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE):
        raise HTTPException(status_code=400, detail=f"image size {h}x{w} outside the accepted range {MIN_SIDE}-{MAX_SIDE} px")
    if not np.isfinite(vol).all():
        raise HTTPException(status_code=400, detail="image contains NaN or infinite values")
    return vol


def png_b64(hu: np.ndarray, window: tuple[float, float]) -> str:
    from PIL import Image

    img = (window_to_unit(hu, *window) * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def npy_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode()


def unit(a: np.ndarray, window: tuple[float, float]) -> torch.Tensor:
    return torch.from_numpy(window_to_unit(np.asarray(a, np.float32), *window))[None, None]


# ---------------------------------------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict:
    """Liveness check. Needs no API key so that monitors and the client can test the connection."""
    return dict(status="ok", api_version=API_VERSION, device=str(DEVICE),
                gpu=torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
                models=len(Registry.available()), auth_required=API_KEY is not None,
                uptime_s=round(time.time() - STATS["started"]), requests=STATS["requests"], errors=STATS["errors"])


@app.get("/models", dependencies=[Depends(require_key)])
def models() -> dict:
    """Every trained model the server can use, with the metadata stored in its checkpoint."""
    out = []
    for organ, arch in Registry.available():
        try:
            _, ck = REGISTRY.get(organ, arch)
        except Exception as exc:
            log.warning("skipping %s/%s: %s", organ, arch, exc)
            continue
        out.append(dict(describe(organ, arch, ck), default=(resolve_arch(organ, "auto") == arch)))
    return dict(models=out)


@app.post("/v1/enhance", dependencies=[Depends(require_key)])
def enhance(file: UploadFile = File(..., description="low-dose slice or volume: .dcm, .npy (HU) or .png/.jpg"),
            organ: str = Form("liver"),
            arch: str = Form("auto", description="architecture key (e.g. resunet) or 'auto' = the organ's best"),
            z: int | None = Form(None, description="slice index for a 3-D volume (default: middle slice)"),
            reference: UploadFile | None = File(None, description="optional routine-dose .npy/.dcm of the same slice, to get metrics"),
            return_npy: bool = Form(True, description="include the enhanced HU array as base64 .npy")) -> dict:
    """Enhance ONE slice. For a 3-D volume the neighbouring slices are used as extra context (2.5-D input)."""
    STATS["requests"] += 1
    try:
        if organ not in ORGANS:
            raise HTTPException(status_code=404, detail=f"unknown organ '{organ}'. Choose from {list(ORGANS)}")
        arch = resolve_arch(organ, arch)
        if arch not in ARCHS:
            raise HTTPException(status_code=404, detail=f"unknown architecture '{arch}'. Choose from {list(ARCHS)}")
        try:
            model, ck = REGISTRY.get(organ, arch)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no trained model for {organ}/{arch} on this server (see GET /models)") from None
        window = tuple(ck["hu_window"])

        vol = to_volume(file.filename, read_upload(file), window)
        zi = vol.shape[0] // 2 if z is None else z
        if not 0 <= zi < vol.shape[0]:
            raise HTTPException(status_code=400, detail=f"z={zi} is outside 0..{vol.shape[0] - 1}")

        t0 = time.perf_counter()
        with INFER_LOCK:
            out = enhance_slice(model, ck, vol, zi, DEVICE)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000

        inp = vol[zi]
        resp = dict(model=describe(organ, arch, ck), slice=zi, n_slices=int(vol.shape[0]), shape=list(out.shape),
                    inference_ms=round(ms, 1), metrics=None,
                    input_png_b64=png_b64(inp, window), enhanced_png_b64=png_b64(out, window), reference_png_b64=None)
        if ck.get("data_source") == "synthetic":
            resp["warning"] = "this model was trained on synthetic data - interface test only"
        if return_npy:
            resp["enhanced_npy_b64"] = npy_b64(out)

        if reference is not None:
            ref_vol = to_volume(reference.filename, read_upload(reference), window)
            ref = ref_vol[min(zi, ref_vol.shape[0] - 1)] if ref_vol.shape[0] > 1 else ref_vol[0]
            if ref.shape != inp.shape:
                raise HTTPException(status_code=400, detail=f"reference shape {ref.shape} differs from input {inp.shape}")
            r, i, o = unit(ref, window), unit(inp, window), unit(out, window)
            resp["metrics"] = dict(
                psnr_in=psnr(i, r).item(), psnr_out=psnr(o, r).item(),
                ssim_in=ssim(i, r).item(), ssim_out=ssim(o, r).item(),
                rmse_in=rmse_hu(i, r, window).item(), rmse_out=rmse_hu(o, r, window).item())
            resp["reference_png_b64"] = png_b64(ref, window)
        log.info("enhance %s/%s z=%d %dx%d -> %.0f ms", organ, arch, zi, *out.shape, ms)
        return resp
    except HTTPException:
        STATS["errors"] += 1
        raise
    except Exception:
        STATS["errors"] += 1
        log.exception("enhance failed")
        raise HTTPException(status_code=500, detail="internal error while enhancing (see the server log)") from None

