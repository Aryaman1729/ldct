"""client.py - Python client + command-line tool for the LDCT-Recon inference server.

It does NOT import torch or your model code: it only needs `requests` and `numpy`, so it can run on any computer
that can reach the server over the network.

As a library:
    from client import LDCTClient
    c = LDCTClient("http://127.0.0.1:8000", api_key=None)
    print(c.health())
    r = c.enhance("slice.npy", organ="liver", arch="resunet", z=40, reference="routine_dose.npy")
    r.enhanced        # numpy array, Hounsfield units
    r.metrics         # PSNR / SSIM / RMSE before and after (only if a reference was sent)
    r.meta            # model info, inference time ...

From the command line (one command at a time):
    python client.py health
    python client.py models
    python client.py enhance --file slice.npy --organ liver --arch resunet --z 40 --out enhanced.npy --png enhanced.png
Server address and key can also come from the environment:  LDCT_SERVER  and  LDCT_API_KEY
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests


class LDCTError(RuntimeError):
    """Anything that went wrong talking to the server, with a human-readable message."""


@dataclass
class EnhanceResult:
    enhanced: np.ndarray | None   # (H, W) float32, Hounsfield units (None if return_npy=False)
    png: bytes                    # enhanced slice as an 8-bit PNG (organ window)
    input_png: bytes
    reference_png: bytes | None
    meta: dict                    # model, slice, n_slices, shape, inference_ms, warning ...
    metrics: dict | None          # psnr_in/out, ssim_in/out, rmse_in/out


def _b64(s: str | None) -> bytes | None:
    return base64.b64decode(s) if s else None


def _npy_bytes(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, np.float32), allow_pickle=False)
    return buf.getvalue()


def _prepare(data, z: int | None, context: int, default_name: str):
    """Path or numpy array -> (filename, bytes, z_to_send, (orig_z, orig_n) | None).

    A 3-D volume is cut down to the slice you want plus `context` neighbours on each side, so a 300-slice scan is not
    uploaded just to enhance one slice (the model only looks at the neighbouring slices)."""
    is_path = isinstance(data, (str, os.PathLike))
    if is_path and not str(data).lower().endswith(".npy"):
        p = Path(data)
        return p.name, p.read_bytes(), z, None          # .dcm / .png / .jpg are sent as they are
    arr = np.load(data, mmap_mode="r", allow_pickle=False) if is_path else np.asarray(data)
    name = Path(data).name if is_path else default_name
    if arr.ndim != 3:
        return name, _npy_bytes(arr), z, None
    n = arr.shape[0]
    zz = n // 2 if z is None else int(z)
    if not 0 <= zz < n:
        raise LDCTError(f"z={zz} is outside 0..{n - 1}")
    lo, hi = max(0, zz - context), min(n, zz + context + 1)
    return name, _npy_bytes(arr[lo:hi]), zz - lo, (zz, n)


class LDCTClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 120.0):
        self.base = (base_url or os.environ.get("LDCT_SERVER") or "http://127.0.0.1:8000").rstrip("/")
        self.key = api_key or os.environ.get("LDCT_API_KEY") or None
        self.timeout = timeout
        self.http = requests.Session()

    # -------------------------------------------------------------------------------------------- plumbing
    def _request(self, method: str, path: str, **kw):
        headers = {"X-API-Key": self.key} if self.key else {}
        try:
            r = self.http.request(method, self.base + path, headers=headers, timeout=self.timeout, **kw)
        except requests.ConnectionError:
            raise LDCTError(f"cannot reach the server at {self.base} - is it running (uvicorn server:app)?") from None
        except requests.Timeout:
            raise LDCTError(f"the server at {self.base} did not answer within {self.timeout:.0f} s") from None
        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text[:300]
            if isinstance(detail, list):  # FastAPI validation errors
                detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in detail)
            raise LDCTError(f"server answered {r.status_code}: {detail}")
        return r.json()

    # ------------------------------------------------------------------------------------------------- API
    def health(self) -> dict:
        return self._request("GET", "/health")

    def models(self) -> list[dict]:
        return self._request("GET", "/models")["models"]

    def enhance(self, data, organ: str = "liver", arch: str | None = None, z: int | None = None,
                reference=None, return_npy: bool = True, context: int = 2) -> EnhanceResult:
        """Send a low-dose slice (path to .dcm/.npy/.png, or a numpy array) and get the enhanced slice back.

        For a 3-D .npy volume, `z` is the slice to enhance (default: middle); only z +/- `context` slices are uploaded.
        `reference` (optional) is the routine-dose version of the same slice/volume and makes the server return metrics."""
        name, raw, z_send, orig = _prepare(data, z, context, "slice.npy")
        files = {"file": (name, raw)}
        if reference is not None:
            if isinstance(reference, (str, os.PathLike)) and not str(reference).lower().endswith(".npy"):
                files["reference"] = (Path(reference).name, Path(reference).read_bytes())
            else:
                ref = np.load(reference, mmap_mode="r", allow_pickle=False) if isinstance(reference, (str, os.PathLike)) else np.asarray(reference)
                if ref.ndim == 3:  # only the one slice we compare against
                    ref = ref[orig[0] if orig else (z if z is not None else ref.shape[0] // 2)]
                files["reference"] = ("reference.npy", _npy_bytes(ref))
        form = {"organ": organ, "arch": arch or "auto", "return_npy": str(bool(return_npy)).lower()}
        if z_send is not None:
            form["z"] = str(int(z_send))
        j = self._request("POST", "/v1/enhance", files=files, data=form)
        npy = _b64(j.get("enhanced_npy_b64"))
        meta = {k: j[k] for k in ("model", "slice", "n_slices", "shape", "inference_ms", "warning") if k in j}
        if orig:  # report positions in the ORIGINAL volume, not in the cropped upload
            meta["slice"], meta["n_slices"] = orig
        return EnhanceResult(
            enhanced=np.load(io.BytesIO(npy), allow_pickle=False) if npy else None,
            png=_b64(j["enhanced_png_b64"]), input_png=_b64(j["input_png_b64"]),
            reference_png=_b64(j.get("reference_png_b64")), meta=meta, metrics=j.get("metrics"))


# ------------------------------------------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Command-line client for the LDCT-Recon server.")
    ap.add_argument("--server", default=None, help="default: $LDCT_SERVER or http://127.0.0.1:8000")
    ap.add_argument("--key", default=None, help="API key (default: $LDCT_API_KEY)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health", help="is the server up?")
    sub.add_parser("models", help="list the trained models on the server")
    e = sub.add_parser("enhance", help="enhance one slice")
    e.add_argument("--file", required=True, help="low-dose .dcm / .npy (HU) / .png")
    e.add_argument("--organ", default="liver")
    e.add_argument("--arch", default=None, help="e.g. resunet (default: the organ's best)")
    e.add_argument("--z", type=int, default=None, help="slice index if --file is a volume (default: middle)")
    e.add_argument("--reference", default=None, help="routine-dose .npy/.dcm of the same slice -> prints metrics")
    e.add_argument("--out", default=None, help="save the enhanced HU array here (.npy)")
    e.add_argument("--png", default=None, help="save the enhanced slice as a PNG here")
    a = ap.parse_args()

    c = LDCTClient(a.server, a.key)
    try:
        if a.cmd == "health":
            print(json.dumps(c.health(), indent=2))
        elif a.cmd == "models":
            for m in c.models():
                star = " (default)" if m["default"] else ""
                vp = f"{m['val_psnr']:.2f} dB" if m.get("val_psnr") else "-"
                print(f"{m['organ']:6s} {m['arch']:10s} {m['label']:24s} val PSNR {vp:>9s} | data: {m['data_source']}{star}")
        else:
            r = c.enhance(a.file, a.organ, a.arch, a.z, a.reference, return_npy=bool(a.out))
            m = r.meta["model"]
            print(f"model {m['organ']}/{m['arch']} | slice {r.meta['slice']} of {r.meta['n_slices']} | "
                  f"{r.meta['shape'][0]}x{r.meta['shape'][1]} | {r.meta['inference_ms']} ms on the server")
            if r.meta.get("warning"):
                print("WARNING:", r.meta["warning"])
            if r.metrics:
                k = r.metrics
                print(f"PSNR {k['psnr_in']:.2f} -> {k['psnr_out']:.2f} dB | SSIM {k['ssim_in']:.4f} -> {k['ssim_out']:.4f} | "
                      f"RMSE {k['rmse_in']:.1f} -> {k['rmse_out']:.1f} HU")
            if a.out:
                np.save(a.out, r.enhanced)
                print("saved", a.out)
            if a.png:
                Path(a.png).write_bytes(r.png)
                print("saved", a.png)
    except LDCTError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
