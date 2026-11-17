"""dataset.py - organ profiles + get the TCIA LDCT-and-Projection-data collection and serve it to PyTorch.

Three jobs, in one file:

1. DOWNLOAD  - talk to the TCIA / NBIA REST API (no manual downloading, no login for the open
               chest and liver collections), pick only the *image* series of the Siemens cases that
               have both a full-dose and a low-dose reconstruction, and download them patient by
               patient.  Each patient is converted to a compact cache and the raw DICOM is deleted
               again, so the full dataset never needs hundreds of GB of disk.
2. CONVERT   - DICOM -> Hounsfield units, full/low pairing by slice position, DICOM metadata saved
               per case (meta.json), coarse body mask, int16 .npy volumes.
3. SERVE     - patient-level train/val/test split and a 2.5D paired-patch torch Dataset.

Command line:
    python dataset.py list     --organ chest               # show what the API offers, sizes (no download)
    python dataset.py download --organ chest               # download + convert everything (resumable)
    python dataset.py download --organ liver --max-patients 5      # quick pilot first
    python dataset.py from-folder --organ chest --dicom-dir D:/TCIA/chest   # already downloaded manually

Cache layout (data/cache/<organ>/<case>/): full.npy, low.npy (int16 HU, N x H x W), body.npy, meta.json.

Facts about the collection (TCIA page): only the ~50 Siemens cases per organ include low-dose *images*;
GE cases have full-dose images + projection data only, so they are skipped.  Low dose is simulated
(chest 10 %, abdomen 25 % of routine dose).  Dataset DOI 10.7937/9npb-2637, licence CC BY 4.0
(chest/liver).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import threading
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dataclasses import asdict, dataclass
import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import Dataset


# =============================================================================== organ profiles
COLLECTION = "LDCT-and-Projection-data"
# Where data/, runs/ and results/ live. Override with the LDCT_HOME environment variable (used by the tests).
PROJECT_HOME = Path(os.environ.get("LDCT_HOME", Path(__file__).resolve().parent))


@dataclass(frozen=True)
class OrganConfig:
    key: str
    title: str
    tcia_prefix: str
    dose_fraction: float
    hu_window: tuple
    patch_size: int
    batch_size: int
    epochs: int
    lr: float
    w_l1: float
    w_ssim: float
    w_edge: float
    min_body_fraction: float
    default_arch: str
    description: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hu_window"] = list(self.hu_window)
        return d


ORGANS: dict[str, OrganConfig] = {
    "chest": OrganConfig(
        key="chest", title="Chest", tcia_prefix="C", dose_fraction=0.10,
        hu_window=(-1000.0, 400.0), patch_size=128, batch_size=32, epochs=100, lr=2e-4,
        w_l1=1.0, w_ssim=0.30, w_edge=0.20, min_body_fraction=0.5, default_arch="resunet",
        description="Low-dose lung protocol (10 % of routine dose). Very noisy input and fine structures "
                    "(vessels, nodules): stronger SSIM and edge terms.",
    ),
    "liver": OrganConfig(
        key="liver", title="Liver / Abdomen", tcia_prefix="L", dose_fraction=0.25,
        hu_window=(-160.0, 240.0), patch_size=128, batch_size=32, epochs=100, lr=2e-4,
        w_l1=1.0, w_ssim=0.20, w_edge=0.10, min_body_fraction=0.5, default_arch="resunet",
        description="Contrast-enhanced abdominal CT (25 % of routine dose). Low-contrast liver lesions: "
                    "remove noise without smoothing lesions away.",
    ),
}
ORGAN_KEYS = list(ORGANS)


ROOT = PROJECT_HOME
DEFAULT_API = "https://nbia.cancerimagingarchive.net/nbia-api/services/v4"
COARSE = 8  # body mask is stored at 1/8 resolution
DEFAULT_LOW_RE = r"low|quarter|reduced"
DEFAULT_FULL_RE = r"full|routine|normal"


# =============================================================================== preprocessing helpers
def window_to_unit(hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to the HU window and map to [0, 1]."""
    return (np.clip(hu, lo, hi) - lo) / (hi - lo)


def unit_to_hu(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return x * (hi - lo) + lo


def coarse_body_fraction(slice_hu: np.ndarray, factor: int = COARSE) -> np.ndarray:
    """Fraction of 'inside the patient' pixels per coarse block (used to skip pure-air patches)."""
    sm = ndi.gaussian_filter(slice_hu.astype(np.float32), 1.5)
    mask = ndi.binary_fill_holes(sm > -300)
    h, w = mask.shape
    hh, ww = h // factor * factor, w // factor * factor
    blocks = mask[:hh, :ww].reshape(hh // factor, factor, ww // factor, factor).mean(axis=(1, 3))
    return blocks.astype(np.float16)


def save_case(cache_dir, key: str, full_hu: np.ndarray, low_hu: np.ndarray, meta: dict) -> dict:
    """Write one paired case (full/low HU volumes + body mask + metadata) to the cache."""
    assert full_hu.shape == low_hu.shape and full_hu.ndim == 3, "expected matching (N,H,W) volumes"
    d = Path(cache_dir) / key
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "full.npy", np.clip(np.rint(full_hu), -32768, 32767).astype(np.int16))
    np.save(d / "low.npy", np.clip(np.rint(low_hu), -32768, 32767).astype(np.int16))
    np.save(d / "body.npy", np.stack([coarse_body_fraction(s) for s in full_hu]))
    meta = dict(meta, key=key, n_slices=int(full_hu.shape[0]), shape=[int(full_hu.shape[1]), int(full_hu.shape[2])])
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, d / "meta.json")  # meta.json appears last => a case is complete only if it exists
    return meta


def load_index(cache_dir) -> list[dict]:
    """Metadata of all complete cached cases, sorted by key."""
    cache_dir = Path(cache_dir)
    out = []
    for m in sorted(cache_dir.glob("*/meta.json")) if cache_dir.exists() else []:
        try:
            out.append(json.loads(m.read_text()))
        except Exception:
            pass
    return out


def cache_dir_for(organ: str) -> Path:
    return ROOT / "data" / "cache" / organ


# =============================================================================== TCIA / NBIA API client
class TCIAClient:
    """Minimal client for the public NBIA REST API (https://wiki.cancerimagingarchive.net/x/ZoATBg).

    Endpoints used (v4):  GET {base}/getSeries?Collection=...        -> JSON list of series
                          GET {base}/getImage?SeriesInstanceUID=...  -> ZIP of the series' DICOM files
    """

    def __init__(self, base: str = DEFAULT_API, timeout: int = 300):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.base, self.timeout = base.rstrip("/"), timeout
        self.s = requests.Session()
        retry = Retry(total=6, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        self.s.mount("https://", HTTPAdapter(max_retries=retry))
        self.s.mount("http://", HTTPAdapter(max_retries=retry))
        self.s.headers["User-Agent"] = "ldct-recon-research/1.0"

    def get_series(self, collection: str = COLLECTION) -> list[dict]:
        r = self.s.get(f"{self.base}/getSeries", params={"Collection": collection, "format": "json"}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected getSeries response: {str(data)[:200]}")
        return data

    def download_series(self, series_uid: str, dest_dir: Path, attempts: int = 4) -> int:
        """Stream the series ZIP to disk, verify it, extract into dest_dir. Returns #files."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        zpath = dest_dir.with_suffix(".zip")
        last = None
        for k in range(attempts):
            try:
                with self.s.get(f"{self.base}/getImage", params={"SeriesInstanceUID": series_uid},
                                stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    with open(zpath, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                if not zipfile.is_zipfile(zpath):
                    raise IOError("response is not a ZIP file (server error page or truncated download)")
                with zipfile.ZipFile(zpath) as z:
                    names = [n for n in z.namelist() if not n.endswith("/")]
                    z.extractall(dest_dir)
                zpath.unlink(missing_ok=True)
                return len(names)
            except Exception as exc:  # retry whole series
                last = exc
                time.sleep(3 * (k + 1))
        zpath.unlink(missing_ok=True)
        raise RuntimeError(f"download of series {series_uid} failed after {attempts} attempts: {last}")


# =============================================================================== series selection (API metadata)
def classify_description(desc: str, low_re: str = DEFAULT_LOW_RE, full_re: str = DEFAULT_FULL_RE) -> str | None:
    """'full', 'low' or None for a series description. Projection series are always skipped."""
    d = (desc or "").lower()
    if "proj" in d:
        return None
    if re.search(low_re, d):
        return "low"
    if re.search(full_re, d):
        return "full"
    return None


def plan_downloads(records: list[dict], organ: str, vendor: str = "SIEMENS",
                   low_re: str = DEFAULT_LOW_RE, full_re: str = DEFAULT_FULL_RE) -> dict:
    """Group the API's series list into {patient: {'full': [...], 'low': [...]}} for one organ.

    Keeps image series of patients with this organ's ID prefix and the chosen scanner vendor that have
    BOTH a full-dose and a low-dose series.  Also returns diagnostics so you can check the filters.
    """
    prefix = ORGANS[organ].tcia_prefix.upper()
    desc_count: Counter = Counter()
    by_pid: dict[str, dict[str, list]] = defaultdict(lambda: {"full": [], "low": []})
    for r in records:
        pid = str(r.get("PatientID", ""))
        if not pid.upper().startswith(prefix):
            continue
        desc = str(r.get("SeriesDescription", ""))
        level = classify_description(desc, low_re, full_re)
        desc_count[(desc, str(r.get("Manufacturer", "")), level or "SKIP")] += 1
        if level is None:
            continue
        if vendor and vendor.lower() not in str(r.get("Manufacturer", "")).lower():
            continue
        by_pid[pid][level].append(r)
    plan = {p: v for p, v in sorted(by_pid.items()) if v["full"] and v["low"]}
    n_bytes = sum(float(s.get("FileSize", 0) or 0) for v in plan.values() for lv in v.values() for s in lv)
    return dict(plan=plan, descriptions=desc_count, n_patients=len(plan), approx_gb=n_bytes / 1e9,
                n_series=sum(len(v["full"]) + len(v["low"]) for v in plan.values()))


def print_plan(info: dict, organ: str) -> None:
    print(f"\n=== {ORGANS[organ].title}: series seen in the collection for patient prefix "
          f"'{ORGANS[organ].tcia_prefix}' ===")
    print(f"{'count':>5}  {'decision':8}  {'manufacturer':28}  series description")
    for (desc, man, lvl), n in sorted(info["descriptions"].items(), key=lambda t: (t[0][2], t[0][0])):
        print(f"{n:5d}  {lvl:8}  {man[:28]:28}  {desc}")
    print(f"\n-> {info['n_patients']} patients with full+low image series, {info['n_series']} series, "
          f"~{info['approx_gb']:.1f} GB to download (compressed DICOM sizes as reported by TCIA)")
    if info["n_patients"] == 0:
        print("!! Nothing selected. Check the decision column: adjust --vendor / --low-regex / --full-regex.")


# =============================================================================== DICOM reading + pairing
def _pydicom():
    import pydicom  # lazy: the rest of the module works without it

    return pydicom


def _is_candidate(name: str) -> bool:
    n = name.lower()
    return n.endswith(".dcm") or "." not in n


def _z_position(ds, fallback: float) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        return float(ipp[2])
    return float(getattr(ds, "InstanceNumber", fallback))


def _safe(text) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "", str(text)) or "na"


def _f(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def discover_series(root: str, organ: str, low_re=DEFAULT_LOW_RE, full_re=DEFAULT_FULL_RE) -> list[dict]:
    """Walk ``root`` and return usable CT image series (headers only; pixels are read later)."""
    pd = _pydicom()
    prefix = ORGANS[organ].tcia_prefix.upper()
    groups: dict[str, list] = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if not _is_candidate(f):
                continue
            p = os.path.join(dirpath, f)
            try:
                ds = pd.dcmread(p, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if not hasattr(ds, "SeriesInstanceUID"):
                continue
            groups[str(ds.SeriesInstanceUID)].append((_z_position(ds, len(groups[str(ds.SeriesInstanceUID)])), p, ds))
    out = []
    for uid, items in groups.items():
        items.sort(key=lambda t: t[0])
        first = items[0][2]
        pid = str(getattr(first, "PatientID", ""))
        rows, cols = int(getattr(first, "Rows", 0) or 0), int(getattr(first, "Columns", 0) or 0)
        level = classify_description(str(getattr(first, "SeriesDescription", "")), low_re, full_re)
        if (not pid.upper().startswith(prefix) or str(getattr(first, "Modality", "")) != "CT"
                or rows != cols or rows < 256 or level is None or len(items) < 3):
            continue
        cur = [_f(getattr(it[2], "XRayTubeCurrent", None)) for it in items]
        sp = getattr(first, "PixelSpacing", [np.nan, np.nan])
        out.append(dict(
            patient_id=pid, series_uid=uid, level=level,
            description=str(getattr(first, "SeriesDescription", "")),
            kernel=str(getattr(first, "ConvolutionKernel", "")),
            thickness=_f(getattr(first, "SliceThickness", None)),
            vendor=str(getattr(first, "Manufacturer", "")), model=str(getattr(first, "ManufacturerModelName", "")),
            kvp=_f(getattr(first, "KVP", None)),
            tube_current_ma=float(np.nanmean(cur)) if np.isfinite(cur).any() else float("nan"),
            pixel_spacing=[_f(sp[0]), _f(sp[1])], z=[it[0] for it in items], files=[it[1] for it in items]))
    return out


def pair_series(series: list[dict]) -> list[tuple[dict, dict]]:
    """Pair each full-dose series with a low-dose series of the same patient (same thickness, similar kernel)."""
    by_pid: dict[str, dict[str, list]] = defaultdict(lambda: {"full": [], "low": []})
    for s in series:
        by_pid[s["patient_id"]][s["level"]].append(s)
    pairs = []
    for pid in sorted(by_pid):
        used: set[str] = set()
        for f in by_pid[pid]["full"]:
            cands = [l for l in by_pid[pid]["low"]
                     if l["series_uid"] not in used and abs(l["thickness"] - f["thickness"]) < 0.11]
            cands.sort(key=lambda l: (l["kernel"] != f["kernel"], abs(len(l["files"]) - len(f["files"]))))
            if cands:
                used.add(cands[0]["series_uid"])
                pairs.append((f, cands[0]))
    return pairs


def read_hu(path: str) -> np.ndarray:
    ds = _pydicom().dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    hu = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    return np.clip(hu, -1024.0, 3071.0)  # also clips padding values such as -2000


def load_pair(full: dict, low: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load slices matched by z-position as HU volumes (N, H, W)."""
    zf, zl = np.asarray(full["z"]), np.asarray(low["z"])
    tol = max(0.25 * (full["thickness"] if np.isfinite(full["thickness"]) else 1.0), 0.2)
    idx = np.abs(zl[None, :] - zf[:, None]).argmin(axis=1)
    keep = np.abs(zl[idx] - zf) <= tol
    if keep.sum() < 3:
        raise ValueError("fewer than 3 slices matched between full and low dose series")
    sel = np.where(keep)[0]
    return (np.stack([read_hu(full["files"][i]) for i in sel]), np.stack([read_hu(low["files"][idx[i]]) for i in sel]))


def convert_folder(folder: str, organ: str, cache_dir, thickness: float | None = None,
                   low_re=DEFAULT_LOW_RE, full_re=DEFAULT_FULL_RE, log=print) -> list[dict]:
    """Find full/low pairs in ``folder`` (any layout) and write them to the cache."""
    cfg = ORGANS[organ]
    pairs = pair_series(discover_series(folder, organ, low_re, full_re))
    if thickness is not None:
        pairs = [(f, l) for f, l in pairs if abs(f["thickness"] - thickness) < 0.11]
    n_per_pid = Counter(f["patient_id"] for f, _ in pairs)
    metas = []
    for f, l in pairs:
        pid = f["patient_id"]
        key = pid if n_per_pid[pid] == 1 else f"{pid}_{f['thickness']:.1f}mm_{_safe(f['kernel'])}"
        if (Path(cache_dir) / key / "meta.json").exists():
            continue
        f_vol, l_vol = load_pair(f, l)
        meta = dict(patient_id=pid, organ=organ, source="tcia", dose_fraction=cfg.dose_fraction,
                    vendor=f["vendor"], scanner_model=f["model"], kernel_full=f["kernel"], kernel_low=l["kernel"],
                    thickness_mm=f["thickness"], kvp=f["kvp"], tube_current_ma=f["tube_current_ma"],
                    pixel_spacing_mm=f["pixel_spacing"], full_series_description=f["description"],
                    low_series_description=l["description"], full_series_uid=f["series_uid"], low_series_uid=l["series_uid"])
        metas.append(save_case(cache_dir, key, f_vol, l_vol, meta))
        log(f"  cached {key}: {f_vol.shape[0]} slices, {f['thickness']} mm, kernel {f['kernel']}")
    return metas


# =============================================================================== full download pipeline
def download_dataset(organ: str, raw_root=None, vendor: str = "SIEMENS", max_patients: int | None = None,
                     thickness: float | None = None, workers: int = 3, keep_raw: bool = False,
                     assume_yes: bool = False, api: str = DEFAULT_API, low_re=DEFAULT_LOW_RE,
                     full_re=DEFAULT_FULL_RE, log=print) -> list[dict]:
    """List -> (confirm) -> per patient: download full+low series, convert, delete raw.  Resumable."""
    cache = cache_dir_for(organ)
    raw_root = Path(raw_root or ROOT / "data" / "raw_tmp" / organ)
    client = TCIAClient(api)
    log(f"Querying {api} for collection '{COLLECTION}' ...")
    info = plan_downloads(client.get_series(), organ, vendor, low_re, full_re)
    print_plan(info, organ)
    plan = info["plan"]
    if not plan:
        raise SystemExit(1)
    pids = list(plan)[:max_patients] if max_patients else list(plan)
    done = {m["patient_id"] for m in load_index(cache)}
    todo = [p for p in pids if p not in done]
    log(f"{len(pids)} patients selected, {len(done & set(pids))} already cached, {len(todo)} to download.")
    if todo and not assume_yes and sys.stdin.isatty():
        if input("Start download? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("aborted")
    lock = threading.Lock()

    def one(pid: str) -> list[dict]:
        folder = raw_root / pid
        try:
            for lv in ("full", "low"):
                for s in plan[pid][lv]:
                    client.download_series(s["SeriesInstanceUID"], folder / s["SeriesInstanceUID"])
            return convert_folder(str(folder), organ, cache, thickness, low_re, full_re,
                                  log=lambda m: (lock.acquire(), log(m), lock.release()))
        finally:
            if not keep_raw:
                shutil.rmtree(folder, ignore_errors=True)

    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(one, p): p for p in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            pid = futs[fu]
            try:
                fu.result()
                log(f"[{i}/{len(todo)}] {pid} done ({(time.time() - t0) / 60:.1f} min elapsed)")
            except Exception as exc:
                failed.append(pid)
                log(f"[{i}/{len(todo)}] {pid} FAILED: {exc}")
    if failed:
        log(f"\n{len(failed)} patients failed ({', '.join(failed)}). Run the same command again to retry them.")
    metas = load_index(cache)
    log(f"Cache now holds {len(metas)} cases / {sum(m['n_slices'] for m in metas)} slice pairs in {cache}")
    return metas


# =============================================================================== split + torch Dataset
def get_splits(cache_dir, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> dict:
    """Patient-level split, stored once in <cache>/splits.json so every model uses the SAME split.

    Never splits by slice (adjacent slices are nearly identical => leakage).  A patient with several
    cases (e.g. 1 mm and 3 mm series) keeps all of them in the same split.  The file is rebuilt if the
    set of cached patients changed.
    """
    cache_dir = Path(cache_dir)
    entries = load_index(cache_dir)
    by_patient: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_patient[e["patient_id"]].append(e["key"])
    pids = sorted(by_patient)
    path = cache_dir / "splits.json"
    if path.exists():
        sp = json.loads(path.read_text())
        if sorted(sum(sp["patients"].values(), [])) == pids:
            return sp
    if len(pids) < 3:
        raise ValueError(f"Need at least 3 patients for train/val/test, found {len(pids)}.")
    random.Random(seed).shuffle(pids)
    n_test, n_val = max(1, round(len(pids) * test_frac)), max(1, round(len(pids) * val_frac))
    groups = {"train": pids[:len(pids) - n_val - n_test], "val": pids[len(pids) - n_val - n_test:len(pids) - n_test],
              "test": pids[len(pids) - n_test:]}
    sp = {k: [key for p in v for key in by_patient[p]] for k, v in groups.items()}
    sp["patients"] = groups
    path.write_text(json.dumps(sp, indent=2))
    return sp


class PairedPatchDataset(Dataset):
    """Random (low-dose slice stack, full-dose centre slice) patch pairs from the cache.

    * ``in_slices`` neighbouring low-dose slices are stacked as channels (2.5D input).
    * Patches that lie mostly outside the body are rejected.
    * Volumes are memory-mapped lazily inside each DataLoader worker (cheap to pickle, low RAM).
    * ``deterministic=True`` makes sample i reproducible (validation).
    """

    def __init__(self, cache_dir, keys: list[str], cfg: OrganConfig, patch_size: int, length: int,
                 in_slices: int = 3, augment: bool = True, deterministic: bool = False, seed: int = 0):
        self.cfg, self.P, self.length = cfg, patch_size, length
        self.augment, self.deterministic, self.seed = augment, deterministic, seed
        self.offsets = np.arange(in_slices) - in_slices // 2
        self.dirs, self.body, counts = [], [], []
        for k in keys:
            d = Path(cache_dir) / k
            shape = np.load(d / "full.npy", mmap_mode="r").shape
            if min(shape[1:]) < patch_size:
                raise ValueError(f"patch_size {patch_size} larger than image {shape[1:]}")
            self.dirs.append(str(d))
            self.body.append(np.load(d / "body.npy"))
            counts.append(shape[0])
        if not self.dirs:
            raise ValueError("Dataset has no cases")
        self.cum = np.concatenate([[0], np.cumsum(counts)])
        self.total = int(self.cum[-1])
        self._mm: dict = {}

    def __len__(self):
        return self.length

    def _vols(self, ci: int):
        if ci not in self._mm:
            d = self.dirs[ci]
            self._mm[ci] = (np.load(d + "/full.npy", mmap_mode="r"), np.load(d + "/low.npy", mmap_mode="r"))
        return self._mm[ci]

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i]) if self.deterministic else np.random.default_rng()
        P, (lo, hi) = self.P, self.cfg.hu_window
        for _ in range(20):
            g = int(rng.integers(self.total))
            ci = int(np.searchsorted(self.cum, g, side="right") - 1)
            z = g - int(self.cum[ci])
            full, low = self._vols(ci)
            H, W = full.shape[1:]
            y0, x0 = int(rng.integers(0, H - P + 1)), int(rng.integers(0, W - P + 1))
            blk = self.body[ci][z, y0 // COARSE:(y0 + P) // COARSE, x0 // COARSE:(x0 + P) // COARSE]
            if blk.size and float(blk.mean()) >= self.cfg.min_body_fraction:
                break
        zs = np.clip(z + self.offsets, 0, full.shape[0] - 1)
        x = window_to_unit(np.asarray(low[zs, y0:y0 + P, x0:x0 + P], np.float32), lo, hi)
        y = window_to_unit(np.asarray(full[z:z + 1, y0:y0 + P, x0:x0 + P], np.float32), lo, hi)
        if self.augment:
            if rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            if rng.random() < 0.5:
                x, y = x[:, ::-1, :], y[:, ::-1, :]
            k = int(rng.integers(4))
            if k:
                x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
        return {"low": torch.from_numpy(np.ascontiguousarray(x)), "full": torch.from_numpy(np.ascontiguousarray(y))}


# =============================================================================== command line
def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["list", "download", "from-folder"])
    p.add_argument("--organ", required=True, choices=list(ORGANS))
    p.add_argument("--vendor", default="SIEMENS", help="only series from this manufacturer ('' = all)")
    p.add_argument("--max-patients", type=int, help="limit patients (pilot run)")
    p.add_argument("--thickness", type=float, help="keep only series of this slice thickness (mm), applied after download")
    p.add_argument("--workers", type=int, default=3, help="patients processed in parallel")
    p.add_argument("--keep-raw", action="store_true", help="do not delete downloaded DICOM after conversion")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.add_argument("--api", default=DEFAULT_API, help="NBIA REST base URL")
    p.add_argument("--low-regex", default=DEFAULT_LOW_RE, help="series-description regex for low-dose series")
    p.add_argument("--full-regex", default=DEFAULT_FULL_RE, help="series-description regex for full-dose series")
    p.add_argument("--dicom-dir", help="from-folder: folder with manually downloaded DICOM")
    a = p.parse_args()
    if a.command == "list":
        print_plan(plan_downloads(TCIAClient(a.api).get_series(), a.organ, a.vendor, a.low_regex, a.full_regex), a.organ)
    elif a.command == "download":
        download_dataset(a.organ, vendor=a.vendor, max_patients=a.max_patients, thickness=a.thickness, workers=a.workers,
                         keep_raw=a.keep_raw, assume_yes=a.yes, api=a.api, low_re=a.low_regex, full_re=a.full_regex)
    else:
        if not a.dicom_dir:
            raise SystemExit("--dicom-dir is required for from-folder")
        metas = convert_folder(a.dicom_dir, a.organ, cache_dir_for(a.organ), a.thickness, a.low_regex, a.full_regex)
        print(f"Cached {len(metas)} new cases in {cache_dir_for(a.organ)}")
        if not metas:
            print("No full/low pairs found - check the folder, the organ and the series descriptions.")


if __name__ == "__main__":
    _cli()