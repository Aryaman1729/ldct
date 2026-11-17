"""model.py - architectures, losses, GPU training, evaluation and the architecture benchmark.

Six denoising architectures share one interface (low-dose stack (B, k, H, W) in [0, 1] -> routine-dose centre slice):

    key         architecture                                   reference
    ----------  ---------------------------------------------  ----------------------------------------
    cnn         3-layer CNN (SRCNN-style direct mapping)        Dong et al., TPAMI 2016 (baseline)
    dncnn       DnCNN, 17 layers, BatchNorm, residual noise     Zhang et al., TIP 2017
    redcnn      RED-CNN, 5 conv + 5 deconv + shortcuts          Chen et al., IEEE TMI 2017 (LDCT-specific)
    unet        U-Net (conv blocks, global residual)            Ronneberger et al., MICCAI 2015
    resunet     Residual U-Net (res blocks, GroupNorm, SiLU)    this project (proposed)
    restormer   Restormer-lite (channel-attention transformer)  Zamir et al., CVPR 2022 (reduced size)

Command line (GPU is used automatically):
    python model.py train   --organ chest --arch auto --epochs 100      # arch 'auto' = winner of the comparison
    python model.py compare --organ chest --epochs 40                   # trains all six, writes results/chest/
    python model.py train   --organ liver --arch redcnn --resume

Training: mixed precision (bf16 / fp16), TF32, cuDNN autotune, channels-last, pinned multi-worker loading, optional
torch.compile, resume, early stopping, graceful stop (create a file STOP in runs/<organ>/<arch>/).
Outputs: runs/<organ>/<arch>/{best.pt,last.pt,history.jsonl,status.json,test_metrics.json}; results/<organ>/ from `compare`
(comparison.md/.csv/_latex.tex/.json, comparison_bars.png, comparison_visual.png, significance.json, best_arch.json).
Model selection uses the validation set; the test set is only used for the reported numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (ORGANS, PROJECT_HOME, PairedPatchDataset, cache_dir_for, get_splits, load_index, unit_to_hu,
                     window_to_unit)

# ====================================================================================================
# architectures
# ====================================================================================================

def _centre(x: torch.Tensor, k: int) -> torch.Tensor:
    return x[:, k // 2: k // 2 + 1]


def _pad_to_multiple(x: torch.Tensor, m: int):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


# ------------------------------------------------------------------------------------------------ 1. CNN
class SimpleCNN(nn.Module):
    """Three convolution layers (9x9 -> 5x5 -> 5x5), direct mapping - the 'plain CNN' baseline."""

    def __init__(self, in_slices: int = 3, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_slices, width, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(width, width // 2, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, 1, 5, padding=2))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------------------------- 2. DnCNN
class DnCNN(nn.Module):
    def __init__(self, in_slices: int = 3, width: int = 64, depth: int = 17):
        super().__init__()
        self.k = in_slices
        layers = [nn.Conv2d(in_slices, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return _centre(x, self.k) + self.net(x)  # network predicts the noise/artefact residual


# ---------------------------------------------------------------------------------------------- 3. RED-CNN
class REDCNN(nn.Module):
    """Residual Encoder-Decoder CNN (Chen et al. 2017): 5 conv + 5 deconv (5x5, no padding), ReLU,
    three residual shortcuts (conv2->deconv3, conv4->deconv1, input->output)."""

    def __init__(self, in_slices: int = 3, width: int = 96):
        super().__init__()
        self.k = in_slices
        self.c1, self.c2, self.c3, self.c4, self.c5 = (
            nn.Conv2d(in_slices, width, 5), nn.Conv2d(width, width, 5), nn.Conv2d(width, width, 5),
            nn.Conv2d(width, width, 5), nn.Conv2d(width, width, 5))
        self.d1, self.d2, self.d3, self.d4 = (nn.ConvTranspose2d(width, width, 5) for _ in range(4))
        self.d5 = nn.ConvTranspose2d(width, 1, 5)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        r1 = _centre(x, self.k)
        o = self.relu(self.c1(x))
        o = self.relu(self.c2(o)); r2 = o
        o = self.relu(self.c3(o))
        o = self.relu(self.c4(o)); r3 = o
        o = self.relu(self.c5(o))
        o = self.d1(o) + r3
        o = self.d2(self.relu(o))
        o = self.d3(self.relu(o)) + r2
        o = self.d4(self.relu(o))
        o = self.d5(self.relu(o)) + r1
        return o


# ------------------------------------------------------------------------------------------------ 4. U-Net
class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        g = lambda c: nn.GroupNorm(min(8, c), c)  # noqa: E731
        self.b = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), g(cout), nn.ReLU(inplace=True),
                               nn.Conv2d(cout, cout, 3, padding=1), g(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b(x)


class UNet(nn.Module):
    """Classic U-Net: max-pool encoder, transposed-conv decoder, concatenation skips; plus a global
    residual (output = centre slice + prediction), as is standard for CT denoising."""

    def __init__(self, in_slices: int = 3, width: int = 32, depth: int = 3):
        super().__init__()
        self.k, self.depth = in_slices, depth
        ch = [width * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_DoubleConv(in_slices if i == 0 else ch[i - 1], ch[i]) for i in range(depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_DoubleConv(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        c = _centre(x, self.k)
        x, h0, w0 = _pad_to_multiple(x, 2 ** self.depth)
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return c + self.head(x)[..., :h0, :w0]


# --------------------------------------------------------------------------------------- 5. Residual U-Net
class _ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(cin, cout, 3, padding=1), nn.Conv2d(cout, cout, 3, padding=1)
        self.n1, self.n2 = nn.GroupNorm(min(8, cout), cout), nn.GroupNorm(min(8, cout), cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        return self.act(self.n2(self.c2(h)) + self.skip(x))


class ResUNet(nn.Module):
    """Proposed: U-Net with residual blocks, GroupNorm, SiLU, strided-conv down-sampling, bilinear
    up-sampling and a zero-initialised output layer (starts as the identity) + global residual."""

    def __init__(self, in_slices: int = 3, width: int = 32, depth: int = 3):
        super().__init__()
        self.k, self.depth = in_slices, depth
        ch = [width * 2 ** i for i in range(depth + 1)]
        self.stem = nn.Conv2d(in_slices, ch[0], 3, padding=1)
        self.enc = nn.ModuleList([_ResBlock(ch[i], ch[i]) for i in range(depth)])
        self.down = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1) for i in range(depth)])
        self.mid = nn.Sequential(_ResBlock(ch[depth], ch[depth]), _ResBlock(ch[depth], ch[depth]))
        self.up = nn.ModuleList([nn.Conv2d(ch[i + 1], ch[i], 3, padding=1) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_ResBlock(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        c = _centre(x, self.k)
        x, h0, w0 = _pad_to_multiple(x, 2 ** self.depth)
        h, skips = self.stem(x), []
        for blk, down in zip(self.enc, self.down):
            h = blk(h)
            skips.append(h)
            h = down(h)
        h = self.mid(h)
        for up, blk in zip(self.up, self.dec):
            h = blk(torch.cat([up(F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)), skips.pop()], 1))
        return c + self.head(h)[..., :h0, :w0]


# ------------------------------------------------------------------------------------------ 6. Restormer-lite
class _LN2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _MDTA(nn.Module):
    """Multi-Dconv-head Transposed Attention: attention across channels (cost linear in pixels)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.t = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.dw(self.qkv(x)).chunk(3, 1)
        q, k, v = (t.reshape(b, self.h, c // self.h, h * w) for t in (q, k, v))
        att = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.t
        return self.proj((att.softmax(-1) @ v).reshape(b, c, h, w))


class _GDFN(nn.Module):
    def __init__(self, dim, expansion=2.0):
        super().__init__()
        hid = int(dim * expansion)
        self.pin = nn.Conv2d(dim, hid * 2, 1, bias=False)
        self.dw = nn.Conv2d(hid * 2, hid * 2, 3, padding=1, groups=hid * 2, bias=False)
        self.pout = nn.Conv2d(hid, dim, 1, bias=False)

    def forward(self, x):
        a, b = self.dw(self.pin(x)).chunk(2, 1)
        return self.pout(F.gelu(a) * b)


class _TBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.att, self.n2, self.ff = _LN2d(dim), _MDTA(dim, heads), _LN2d(dim), _GDFN(dim)

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.ff(self.n2(x))


class RestormerLite(nn.Module):
    """Three-level Restormer-style encoder-decoder (blocks [2,2,2] / bottleneck 3), channel attention."""

    def __init__(self, in_slices: int = 3, width: int = 32, blocks=(2, 2, 2), mid_blocks: int = 3):
        super().__init__()
        self.k, d = in_slices, len(blocks)
        ch = [width * 2 ** i for i in range(d + 1)]
        heads = [max(1, 2 ** i) for i in range(d + 1)]
        self.embed = nn.Conv2d(in_slices, ch[0], 3, padding=1)
        self.enc = nn.ModuleList([nn.Sequential(*[_TBlock(ch[i], heads[i]) for _ in range(blocks[i])]) for i in range(d)])
        self.down = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1) for i in range(d)])
        self.mid = nn.Sequential(*[_TBlock(ch[d], heads[d]) for _ in range(mid_blocks)])
        self.up = nn.ModuleList([nn.Conv2d(ch[i + 1], ch[i], 1) for i in reversed(range(d))])
        self.fuse = nn.ModuleList([nn.Conv2d(2 * ch[i], ch[i], 1) for i in reversed(range(d))])
        self.dec = nn.ModuleList([nn.Sequential(*[_TBlock(ch[i], heads[i]) for _ in range(blocks[i])]) for i in reversed(range(d))])
        self.out = nn.Conv2d(ch[0], 1, 3, padding=1)
        self.depth = d

    def forward(self, x):
        c = _centre(x, self.k)
        x, h0, w0 = _pad_to_multiple(x, 2 ** self.depth)
        h, skips = self.embed(x), []
        for blk, down in zip(self.enc, self.down):
            h = blk(h)
            skips.append(h)
            h = down(h)
        h = self.mid(h)
        for up, fuse, blk in zip(self.up, self.fuse, self.dec):
            h = up(F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False))
            h = blk(fuse(torch.cat([h, skips.pop()], 1)))
        return c + self.out(h)[..., :h0, :w0]


# --------------------------------------------------------------------------------------------- registry
ARCHS = {
    "cnn": dict(cls=SimpleCNN, width=64, label="CNN (3-layer)", ref="Dong et al. 2016"),
    "dncnn": dict(cls=DnCNN, width=64, label="DnCNN", ref="Zhang et al. 2017"),
    "redcnn": dict(cls=REDCNN, width=96, label="RED-CNN", ref="Chen et al. 2017"),
    "unet": dict(cls=UNet, width=32, label="U-Net", ref="Ronneberger et al. 2015"),
    "resunet": dict(cls=ResUNet, width=32, label="Residual U-Net (ours)", ref="this work"),
    "restormer": dict(cls=RestormerLite, width=32, label="Restormer-lite", ref="Zamir et al. 2022"),
}


def build_model(arch: str, in_slices: int = 3, width: int | None = None) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"unknown architecture '{arch}'. Choose from {list(ARCHS)}")
    info = ARCHS[arch]
    return info["cls"](in_slices=in_slices, width=width or info["width"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ====================================================================================================
# losses and metrics
# ====================================================================================================

def _gauss_kernel(win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    ax = torch.arange(win, dtype=torch.float32) - (win - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, win: int = 11,
         sigma: float = 1.5, per_image: bool = False) -> torch.Tensor:
    """Structural similarity with a Gaussian window (valid convolution)."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    k = _gauss_kernel(win, sigma).to(device=x.device, dtype=x.dtype)
    mu_x, mu_y = F.conv2d(x, k), F.conv2d(y, k)
    sxx = F.conv2d(x * x, k) - mu_x ** 2
    syy = F.conv2d(y * y, k) - mu_y ** 2
    sxy = F.conv2d(x * y, k) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(1, 2, 3)) if per_image else s.mean()


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, per_image: bool = False) -> torch.Tensor:
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    p = 10 * torch.log10(data_range ** 2 / mse)
    return p if per_image else p.mean()


def rmse_hu(x: torch.Tensor, y: torch.Tensor, hu_window: tuple) -> torch.Tensor:
    """RMSE converted back to Hounsfield units (per image)."""
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3))
    return mse.sqrt() * (hu_window[1] - hu_window[0])


def _gradients(t: torch.Tensor):
    return t[..., :, 1:] - t[..., :, :-1], t[..., 1:, :] - t[..., :-1, :]


class CompositeLoss(nn.Module):
    """w_l1 * L1  +  w_ssim * (1 - SSIM)  +  w_edge * L1(gradients).

    L1 alone gives good pixel accuracy but slightly smooth results; the SSIM term
    protects local structure and contrast, and the gradient term keeps edges
    (vessel walls, organ borders, lesion margins) sharp.
    """

    def __init__(self, w_l1: float = 1.0, w_ssim: float = 0.2, w_edge: float = 0.1):
        super().__init__()
        self.w_l1, self.w_ssim, self.w_edge = w_l1, w_ssim, w_edge

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1 = F.l1_loss(pred, target)
        ss = 1.0 - ssim(pred, target)
        pdx, pdy = _gradients(pred)
        tdx, tdy = _gradients(target)
        edge = F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)
        total = self.w_l1 * l1 + self.w_ssim * ss + self.w_edge * edge
        return total, {"l1": l1.detach(), "ssim_loss": ss.detach(), "edge": edge.detach()}


# ====================================================================================================
# inference and evaluation
# ====================================================================================================

def pick_device(name: str | None = None) -> torch.device:
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_model(path, device=None):
    """Load a checkpoint written by training.py -> (model in eval mode, checkpoint dict)."""
    device = device or pick_device()
    ck = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ck["arch"], ck["in_slices"], ck["width"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


@torch.no_grad()
def enhance_slice(model, ck: dict, volume_hu: np.ndarray, z: int, device=None) -> np.ndarray:
    """Enhance slice ``z`` of an (N, H, W) HU volume -> (H, W) HU image inside the organ's HU window."""
    device = device or next(model.parameters()).device
    lo, hi = ck["hu_window"]
    k = ck["in_slices"]
    zs = np.clip(z + np.arange(k) - k // 2, 0, volume_hu.shape[0] - 1)
    x = torch.from_numpy(window_to_unit(volume_hu[zs].astype(np.float32), lo, hi))[None].to(device)
    y = model(x).clamp(0, 1)[0, 0].float().cpu().numpy()
    return unit_to_hu(y, lo, hi)


def _t(a, lo, hi):
    return torch.from_numpy(window_to_unit(np.asarray(a, np.float32), lo, hi))[None, None]


def evaluate_run(run_dir, cache_dir, max_slices: int = 40, device=None, make_figure: bool = True) -> dict:
    """Evaluate best.pt on full test slices (same slices for every model -> paired comparison)."""
    run_dir = Path(run_dir)
    device = device or pick_device()
    model, ck = load_model(run_dir / "best.pt", device)
    lo, hi = ck["hu_window"]
    splits = get_splits(cache_dir)
    rows, example = [], None
    for key in splits["test"]:
        d = Path(cache_dir) / key
        full = np.load(d / "full.npy", mmap_mode="r")
        low = np.load(d / "low.npy")
        body = np.load(d / "body.npy")
        keep = [z for z in range(full.shape[0]) if body[z].mean() > 0.15]
        if not keep:
            continue
        picks = np.linspace(0, len(keep) - 1, min(max_slices, len(keep))).astype(int)
        for n, z in enumerate(keep[i] for i in picks):
            out = enhance_slice(model, ck, low, z, device)
            ref, inp, enh = _t(full[z], lo, hi), _t(low[z], lo, hi), _t(out, lo, hi)
            rows.append(dict(case=key, slice=int(z),
                             psnr_in=psnr(inp, ref).item(), psnr_out=psnr(enh, ref).item(),
                             ssim_in=ssim(inp, ref).item(), ssim_out=ssim(enh, ref).item(),
                             rmse_in=rmse_hu(inp, ref, (lo, hi)).item(), rmse_out=rmse_hu(enh, ref, (lo, hi)).item()))
            if example is None and len(rows) == max(1, len(picks) // 2):
                example = (np.asarray(low[z], np.float32), out, np.asarray(full[z], np.float32), key, z)
    if not rows:
        raise RuntimeError("No test slices evaluated.")
    summary = {k: dict(mean=float(np.mean([r[k] for r in rows])), std=float(np.std([r[k] for r in rows])))
               for k in ["psnr_in", "psnr_out", "ssim_in", "ssim_out", "rmse_in", "rmse_out"]}
    result = dict(organ=ck["organ"], arch=ck["arch"], data_source=ck["data_source"], n_test_cases=len(splits["test"]),
                  n_slices=len(rows), summary=summary, per_slice=rows, best_epoch=ck["epoch"])
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    if make_figure and example is not None:
        _figure(example, (lo, hi), run_dir / "test_example.png", ck["organ"], ck["arch"])
    return result


def _figure(example, window, path, organ, arch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low, out, full, key, z = example
    ref = np.clip(full, *window)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    panels = [("Low dose (input)", low, "gray"), (f"Enhanced ({arch})", out, "gray"),
              ("Routine dose (reference)", full, "gray"), ("|Enhanced - reference|", np.abs(out - ref), "magma")]
    for a, (title, im, cmap) in zip(ax, panels):
        kw = dict(vmin=0, vmax=(window[1] - window[0]) * 0.15) if cmap == "magma" else dict(vmin=window[0], vmax=window[1])
        a.imshow(im, cmap=cmap, **kw)
        a.set_title(title, fontsize=11)
        a.axis("off")
    fig.suptitle(f"{organ} - test case {key}, slice {z}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ====================================================================================================
# training
# ====================================================================================================

ROOT = PROJECT_HOME  # data/, runs/, results/ live here (override: LDCT_HOME)


@dataclass
class TrainArgs:
    organ: str
    arch: str = "auto"                 # a key of ARCHS, or 'auto' = best from `model.py compare`
    cache_dir: str | None = None
    out_dir: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    patch_size: int | None = None
    width: int | None = None
    lr: float | None = None
    patches_per_epoch: int | None = None   # default: 2 x number of training slices (max 40000)
    val_patches: int = 512
    in_slices: int = 3
    patience: int = 20
    seed: int = 42
    num_workers: int | None = None
    device: str | None = None
    amp: bool = True
    compile: bool = False
    resume: bool = False
    skip_test: bool = False
    test_slices: int = 40


def resolve_arch(organ: str, arch: str) -> str:
    if arch != "auto":
        return arch
    f = ROOT / "results" / organ / "best_arch.json"
    if f.exists():
        return json.loads(f.read_text())["arch"]
    return ORGANS[organ].default_arch


def run_dir_for(organ: str, arch: str) -> Path:
    return ROOT / "runs" / organ / arch


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


@torch.no_grad()
def _validate(model, loader, device, hu_window, amp_dtype, baseline=False):
    model.eval()
    ps, ss, rm = [], [], []
    for b in loader:
        low, full = b["low"].to(device, non_blocking=True), b["full"].to(device, non_blocking=True)
        if baseline:
            k = low.shape[1]
            pred = low[:, k // 2: k // 2 + 1]
        else:
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = model(low)
            pred = pred.float().clamp(0, 1)
        ps.append(psnr(pred, full, per_image=True))
        ss.append(ssim(pred, full, per_image=True))
        rm.append(rmse_hu(pred, full, hu_window))
    return torch.cat(ps).mean().item(), torch.cat(ss).mean().item(), torch.cat(rm).mean().item()


def train(args: TrainArgs, log=print) -> dict:
    arch = resolve_arch(args.organ, args.arch)
    args.arch = arch
    out = Path(args.out_dir or run_dir_for(args.organ, arch))
    out.mkdir(parents=True, exist_ok=True)
    (out / "STOP").unlink(missing_ok=True)
    status_path, hist_path = out / "status.json", out / "history.jsonl"
    if not (args.resume and hist_path.exists()):
        hist_path.write_text("")
    status = dict(state="starting", organ=args.organ, arch=arch, epoch=0, started=time.time())
    _write_json(status_path, status)
    try:
        return _train(args, log, out, status, status_path, hist_path)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        _write_json(status_path, status)
        raise


def _train(args: TrainArgs, log, out: Path, status: dict, status_path: Path, hist_path: Path) -> dict:
    cfg, arch = ORGANS[args.organ], args.arch
    cache = Path(args.cache_dir or cache_dir_for(args.organ))
    hp = dict(epochs=args.epochs or cfg.epochs, batch_size=args.batch_size or cfg.batch_size,
              patch_size=args.patch_size or cfg.patch_size, lr=args.lr or cfg.lr,
              width=args.width or ARCHS[arch]["width"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    on_gpu = device.type == "cuda"
    if on_gpu:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu = torch.cuda.get_device_name(device)
    else:
        gpu = "CPU (no CUDA GPU found - training will be slow; see README for the GPU setup)"
    amp_dtype = None
    if args.amp and on_gpu:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    entries = load_index(cache)
    if not entries:
        raise RuntimeError(f"No cached cases in {cache}. Run: python dataset.py download --organ {args.organ}")
    splits = get_splits(cache, args.seed)
    n_train_slices = sum(e["n_slices"] for e in entries if e["key"] in set(splits["train"]))
    ppe = args.patches_per_epoch or min(40000, 2 * n_train_slices)
    hp["patches_per_epoch"] = ppe
    source = entries[0].get("source", "unknown")
    log(f"[{cfg.title} | {arch}] {len(entries)} cases, {n_train_slices} training slices | patients train/val/test = "
        f"{len(splits['patients']['train'])}/{len(splits['patients']['val'])}/{len(splits['patients']['test'])} | "
        f"source: {source}\ndevice: {gpu} | amp: {amp_dtype} | hyper-parameters: {hp}")

    nw = args.num_workers if args.num_workers is not None else min(8, os.cpu_count() or 1)
    common = dict(cache_dir=cache, cfg=cfg, patch_size=hp["patch_size"], in_slices=args.in_slices)
    ds_tr = PairedPatchDataset(keys=splits["train"], length=ppe, augment=True, **common)
    ds_va = PairedPatchDataset(keys=splits["val"], length=args.val_patches, augment=False, deterministic=True,
                               seed=args.seed, **common)
    kw = dict(num_workers=nw, pin_memory=on_gpu, persistent_workers=nw > 0, **({"prefetch_factor": 4} if nw > 0 else {}))
    dl_tr = DataLoader(ds_tr, batch_size=hp["batch_size"], shuffle=False, drop_last=True, **kw)
    dl_va = DataLoader(ds_va, batch_size=hp["batch_size"], shuffle=False, **kw)

    model = build_model(arch, args.in_slices, hp["width"]).to(device)
    if on_gpu:
        model = model.to(memory_format=torch.channels_last)
    n_params = count_parameters(model)
    run_model = torch.compile(model) if args.compile and hasattr(torch, "compile") else model
    log(f"{ARCHS[arch]['label']}: {n_params / 1e6:.2f} M parameters")

    criterion = CompositeLoss(cfg.w_l1, cfg.w_ssim, cfg.w_edge)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    iters = max(1, len(dl_tr))
    total, warm = hp["epochs"] * iters, max(1, min(500, hp["epochs"] * iters // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    b_psnr, b_ssim, b_rmse = _validate(model, dl_va, device, cfg.hu_window, None, baseline=True)
    log(f"Baseline (low-dose input vs routine dose): PSNR {b_psnr:.2f} dB | SSIM {b_ssim:.4f} | RMSE {b_rmse:.1f} HU")

    def ckpt(epoch, vp):
        return dict(model=model.state_dict(), arch=arch, in_slices=args.in_slices, width=hp["width"], organ=args.organ,
                    organ_cfg=cfg.to_dict(), hu_window=list(cfg.hu_window), data_source=source, epoch=epoch,
                    val_psnr=vp, baseline_psnr=b_psnr, hyperparameters=hp, n_params=n_params)

    start, best, bad = 1, -1e9, 0
    if args.resume and (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best, bad = st["epoch"] + 1, st["best"], st["bad"]
        log(f"Resumed from epoch {st['epoch']} (best val PSNR {best:.2f})")

    status.update(state="running", n_params=n_params, hyperparameters=hp, device=gpu, amp=str(amp_dtype), data_source=source,
                  baseline=dict(psnr=b_psnr, ssim=b_ssim, rmse_hu=b_rmse), total_epochs=hp["epochs"],
                  best_psnr=None if best < -1e8 else best, label=ARCHS[arch]["label"])
    _write_json(status_path, status)

    t0, stopped = time.time(), False
    for epoch in range(start, hp["epochs"] + 1):
        model.train()
        acc, n, te = {"loss": 0.0, "l1": 0.0, "ssim_loss": 0.0, "edge": 0.0}, 0, time.time()
        for b in dl_tr:
            low = b["low"].to(device, non_blocking=True)
            full = b["full"].to(device, non_blocking=True)
            if on_gpu:
                low = low.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = run_model(low)
            loss, parts = criterion(pred.float(), full)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            acc["loss"] += loss.item()
            for k, v in parts.items():
                acc[k] += v.item()
            n += 1
            if n % 20 == 0 and (out / "STOP").exists():
                stopped = True
                break
        vp, vs, vr = _validate(model, dl_va, device, cfg.hu_window, amp_dtype)
        row = dict(epoch=epoch, **{k: v / max(1, n) for k, v in acc.items()}, val_psnr=vp, val_ssim=vs, val_rmse_hu=vr,
                   lr=opt.param_groups[0]["lr"], epoch_s=time.time() - te, elapsed_s=time.time() - t0)
        with open(hist_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if vp > best:
            best, bad = vp, 0
            torch.save(ckpt(epoch, vp), out / "best.pt")
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=epoch,
                        best=best, bad=bad), out / "last.pt")
        status.update(epoch=epoch, best_psnr=best, last=row)
        _write_json(status_path, status)
        log(f"epoch {epoch:03d}/{hp['epochs']} | loss {row['loss']:.4f} | val PSNR {vp:.2f} dB (best {best:.2f}) | "
            f"SSIM {vs:.4f} | RMSE {vr:.1f} HU | {row['epoch_s']:.0f}s/epoch")
        if stopped:
            log("Stopped by user.")
            break
        if bad >= args.patience:
            log(f"Early stopping: no validation improvement for {args.patience} epochs.")
            break

    status.update(state="stopped" if stopped else "finished", finished=time.time(), best_psnr=best,
                  train_minutes=(time.time() - t0) / 60)
    if not (out / "best.pt").exists():
        status.update(state="failed", error="no checkpoint was written")
        _write_json(status_path, status)
        return status
    if not args.skip_test:
        res = evaluate_run(out, cache, max_slices=args.test_slices, device=device)
        s = res["summary"]
        status["test"] = dict(n_slices=res["n_slices"], **{k: v["mean"] for k, v in s.items()})
        log(f"TEST ({res['n_test_cases']} patients, {res['n_slices']} slices): PSNR {s['psnr_in']['mean']:.2f} -> "
            f"{s['psnr_out']['mean']:.2f} dB | SSIM {s['ssim_in']['mean']:.4f} -> {s['ssim_out']['mean']:.4f} | "
            f"RMSE {s['rmse_in']['mean']:.1f} -> {s['rmse_out']['mean']:.1f} HU")
    _write_json(status_path, status)
    return status


def add_train_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--arch", default="auto", choices=["auto"] + list(ARCHS),
                   help="auto = best architecture from `model.py compare` (default resunet)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--patch-size", type=int)
    p.add_argument("--width", type=int, help="base channel width of the network")
    p.add_argument("--lr", type=float)
    p.add_argument("--patches-per-epoch", type=int)
    p.add_argument("--in-slices", type=int, default=3, choices=[1, 3, 5])
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, help="data-loader workers (default min(8, CPU cores))")
    p.add_argument("--device", help="cuda, cuda:1, cpu (default: GPU if available)")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--compile", action="store_true", help="torch.compile (faster on PyTorch 2.x + Linux)")
    p.add_argument("--resume", action="store_true", help="continue from runs/<organ>/<arch>/last.pt")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--test-slices", type=int, default=40, help="slices per test patient to evaluate")
    p.add_argument("--cache-dir")
    p.add_argument("--out-dir")


def args_from_namespace(organ: str, a: argparse.Namespace) -> TrainArgs:
    return TrainArgs(organ=organ, arch=a.arch, cache_dir=a.cache_dir, out_dir=a.out_dir, epochs=a.epochs,
                     batch_size=a.batch_size, patch_size=a.patch_size, width=a.width, lr=a.lr,
                     patches_per_epoch=a.patches_per_epoch, in_slices=a.in_slices, patience=a.patience, seed=a.seed,
                     num_workers=a.num_workers, device=a.device, amp=not a.no_amp, compile=a.compile,
                     resume=a.resume, skip_test=a.skip_test, test_slices=a.test_slices)


def train_main(organ: str | None = None) -> None:
    p = argparse.ArgumentParser(description="Train a low-dose CT enhancement model.")
    if organ is None:
        p.add_argument("--organ", required=True, choices=list(ORGANS))
    add_train_arguments(p)
    a = p.parse_args()
    organ = organ or a.organ
    if not load_index(a.cache_dir or cache_dir_for(organ)):
        raise SystemExit(f"No cached data for '{organ}'. Run first:\n  python dataset.py download --organ {organ}")
    train(args_from_namespace(organ, a))


# ====================================================================================================
# architecture comparison
# ====================================================================================================

def _inference_ms(model, ck, device, size=512, reps=10) -> float:
    x = torch.rand(1, ck["in_slices"], size, size, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(reps):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t) / reps * 1000


def collect(organ: str, archs: list[str], device) -> list[dict]:
    rows = []
    for a in archs:
        rd = run_dir_for(organ, a)
        tm = json.loads((rd / "test_metrics.json").read_text())
        st = json.loads((rd / "status.json").read_text())
        model, ck = load_model(rd / "best.pt", device)
        s = tm["summary"]
        rows.append(dict(arch=a, label=ARCHS[a]["label"], params_m=ck["n_params"] / 1e6, best_epoch=ck["epoch"],
                         val_psnr=ck["val_psnr"], train_min=st.get("train_minutes", float("nan")),
                         infer_ms=_inference_ms(model, ck, device),
                         **{k: s[k]["mean"] for k in s}, **{k + "_std": s[k]["std"] for k in s}))
    return rows


def write_tables(rows: list[dict], out: Path, organ: str) -> None:
    (out / "comparison.json").write_text(json.dumps(rows, indent=2))
    base = rows[0]
    head = ["Model", "Params (M)", "PSNR (dB)", "SSIM", "RMSE (HU)", "ms/slice", "Train (min)"]
    lines = [f"| {' | '.join(head)} |", "|" + "---|" * len(head),
             f"| Low-dose input | - | {base['psnr_in']:.2f} | {base['ssim_in']:.4f} | {base['rmse_in']:.1f} | - | - |"]
    tex = ["\\begin{tabular}{lrrrrrr}", "\\hline", " & ".join(head).replace("%", "\\%") + " \\\\", "\\hline",
           f"Low-dose input & -- & {base['psnr_in']:.2f} & {base['ssim_in']:.4f} & {base['rmse_in']:.1f} & -- & -- \\\\"]
    best = max(rows, key=lambda r: r["val_psnr"])
    csv = ["model,params_m,psnr,ssim,rmse_hu,infer_ms,train_min"]
    for r in rows:
        star = "**" if r is best else ""
        lines.append(f"| {star}{r['label']}{star} | {r['params_m']:.2f} | {star}{r['psnr_out']:.2f} ± {r['psnr_out_std']:.2f}{star} | "
                     f"{r['ssim_out']:.4f} | {r['rmse_out']:.1f} | {r['infer_ms']:.0f} | {r['train_min']:.0f} |")
        b = "\\textbf" if r is best else ""
        f = lambda t: f"{b}{{{t}}}" if b else t  # noqa: E731
        tex.append(f"{r['label']} & {r['params_m']:.2f} & {f(format(r['psnr_out'], '.2f'))} & {f(format(r['ssim_out'], '.4f'))} & "
                   f"{f(format(r['rmse_out'], '.1f'))} & {r['infer_ms']:.0f} & {r['train_min']:.0f} \\\\")
        csv.append(f"{r['arch']},{r['params_m']:.3f},{r['psnr_out']:.3f},{r['ssim_out']:.5f},{r['rmse_out']:.2f},"
                   f"{r['infer_ms']:.1f},{r['train_min']:.1f}")
    tex += ["\\hline", "\\end{tabular}"]
    (out / "comparison.md").write_text(f"### {ORGANS[organ].title}: architecture comparison (test set, mean ± std over slices)\n\n"
                                       + "\n".join(lines) + "\n")
    (out / "comparison_latex.tex").write_text("\n".join(tex) + "\n")
    (out / "comparison.csv").write_text("\n".join(csv) + "\n")


def bars(rows, out: Path, organ: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, (k, title, better) in zip(ax, [("psnr_out", "PSNR (dB)  higher = better", 1), ("ssim_out", "SSIM  higher = better", 1),
                                         ("rmse_out", "RMSE (HU)  lower = better", -1)]):
        vals = [r[k] for r in rows]
        err = [r[k + "_std"] for r in rows]
        idx = int(np.argmax(vals) if better > 0 else np.argmin(vals))
        cols = ["#1f7a8c" if i == idx else "#9db4c0" for i in range(len(rows))]
        a.bar(range(len(rows)), vals, yerr=err, color=cols, capsize=3)
        a.axhline(rows[0][k.replace("_out", "_in")], color="#b5483a", ls="--", lw=1.2, label="low-dose input")
        a.set_xticks(range(len(rows)), labels, rotation=30, ha="right", fontsize=8)
        lo_, hi_ = min(vals) - max(err) * 1.5, max(vals) + max(err) * 1.5
        if k != "rmse_out":
            lo_ = min(lo_, rows[0][k.replace("_out", "_in")] - 0.01 * abs(hi_))
        a.set_ylim(lo_, hi_ + 0.08 * (hi_ - lo_))
        a.set_title(title, fontsize=10)
        a.legend(fontsize=8)
    fig.suptitle(f"{ORGANS[organ].title}: architecture comparison on held-out test patients")
    fig.tight_layout()
    fig.savefig(out / "comparison_bars.png", dpi=150)
    plt.close(fig)


def qualitative(organ: str, archs: list[str], out: Path, device) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = cache_dir_for(organ)
    key = get_splits(cache)["test"][0]
    full, low = np.load(cache / key / "full.npy"), np.load(cache / key / "low.npy")
    body = np.load(cache / key / "body.npy")
    z = [i for i in range(len(full)) if body[i].mean() > 0.15][len([i for i in range(len(full)) if body[i].mean() > 0.15]) // 2]
    lo, hi = ORGANS[organ].hu_window
    imgs = [("Low dose", low[z].astype(np.float32)), ("Routine dose (ref.)", full[z].astype(np.float32))]
    for a in archs:
        m, ck = load_model(run_dir_for(organ, a) / "best.pt", device)
        imgs.append((ARCHS[a]["label"], enhance_slice(m, ck, low, z, device)))
    ref = np.clip(imgs[1][1], lo, hi)
    H, W = ref.shape
    y0, x0, s = H // 2 - H // 8, W // 2 - W // 8, H // 4  # central zoom ROI
    n = len(imgs)
    fig, ax = plt.subplots(3, n, figsize=(2.6 * n, 8))
    for j, (t, im) in enumerate(imgs):
        ax[0, j].imshow(im, cmap="gray", vmin=lo, vmax=hi)
        ax[0, j].add_patch(plt.Rectangle((x0, y0), s, s, fill=False, ec="yellow", lw=1))
        ax[0, j].set_title(t, fontsize=8)
        ax[1, j].imshow(im[y0:y0 + s, x0:x0 + s], cmap="gray", vmin=lo, vmax=hi)
        ax[2, j].imshow(np.abs(np.clip(im, lo, hi) - ref), cmap="magma", vmin=0, vmax=(hi - lo) * 0.12)
        for r in range(3):
            ax[r, j].axis("off")
    ax[1, 0].set_title("zoomed ROI", fontsize=8, loc="left")
    ax[2, 0].set_title("abs. error vs reference", fontsize=8, loc="left")
    fig.suptitle(f"{ORGANS[organ].title} - test case {key}, slice {z}")
    fig.tight_layout()
    fig.savefig(out / "comparison_visual.png", dpi=150)
    plt.close(fig)


def significance(organ: str, archs: list[str], best: str, out: Path) -> dict:
    from scipy.stats import wilcoxon

    per = {a: json.loads((run_dir_for(organ, a) / "test_metrics.json").read_text())["per_slice"] for a in archs}
    res = {}
    for a in archs:
        if a == best:
            continue
        x = np.array([r["psnr_out"] for r in per[best]])
        y = np.array([r["psnr_out"] for r in per[a]])
        if len(x) == len(y) and len(x) > 5 and np.any(x != y):
            st = wilcoxon(x, y)
            res[a] = dict(mean_psnr_gain_db=float((x - y).mean()), p_value=float(st.pvalue), n=len(x))
    (out / "significance.json").write_text(json.dumps(dict(reference=best, vs=res), indent=2))
    return res


def compare_main() -> None:
    p = argparse.ArgumentParser(description="Train + compare all architectures for one organ.")
    p.add_argument("--organ", required=True, choices=list(ORGANS))
    p.add_argument("--archs", nargs="+", default=list(ARCHS), choices=list(ARCHS))
    p.add_argument("--epochs", type=int, default=40, help="same budget for every architecture")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--patch-size", type=int)
    p.add_argument("--patches-per-epoch", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-slices", type=int, default=40, help="test slices per patient")
    p.add_argument("--reuse", action="store_true", help="skip architectures that already finished training")
    p.add_argument("--report-only", action="store_true", help="only rebuild tables/figures from existing runs")
    a = p.parse_args()

    device = pick_device(a.device)
    for arch in a.archs:
        rd = run_dir_for(a.organ, arch)
        done = (rd / "test_metrics.json").exists() and (rd / "status.json").exists() and \
            json.loads((rd / "status.json").read_text()).get("state") in ("finished", "stopped")
        if a.report_only or (a.reuse and done):
            print(f"== {arch}: using existing run")
            continue
        print(f"\n===== training {ARCHS[arch]['label']} ({a.organ}) =====")
        train(TrainArgs(organ=a.organ, arch=arch, epochs=a.epochs, batch_size=a.batch_size, patch_size=a.patch_size,
                        patches_per_epoch=a.patches_per_epoch, num_workers=a.num_workers, patience=a.patience,
                        device=a.device, seed=a.seed, test_slices=a.test_slices))
    missing = [x for x in a.archs if not (run_dir_for(a.organ, x) / "test_metrics.json").exists()]
    if missing:
        raise SystemExit(f"No finished run for: {missing}")
    out = ROOT / "results" / a.organ
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(a.organ, a.archs, device)
    best = max(rows, key=lambda r: r["val_psnr"])
    write_tables(rows, out, a.organ)
    bars(rows, out, a.organ)
    qualitative(a.organ, a.archs, out, device)
    sig = significance(a.organ, a.archs, best["arch"], out)
    (out / "best_arch.json").write_text(json.dumps(dict(arch=best["arch"], label=best["label"], selected_by="validation PSNR",
                                                        val_psnr=best["val_psnr"], epochs=a.epochs), indent=2))
    print("\n" + (out / "comparison.md").read_text())
    print(f"Best architecture for {a.organ} (by validation PSNR): {best['label']}")
    for k, v in sig.items():
        print(f"  vs {ARCHS[k]['label']}: +{v['mean_psnr_gain_db']:.2f} dB, Wilcoxon p = {v['p_value']:.2g}")
    print(f"Results in {out}")


# ====================================================================================================================
# command line:  python model.py train|compare ...
# ====================================================================================================================
def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "compare"):
        raise SystemExit("usage: python model.py train --organ {chest,liver} [options]\n"
                         "       python model.py compare --organ {chest,liver} [options]")
    cmd = sys.argv.pop(1)
    train_main() if cmd == "train" else compare_main()


if __name__ == "__main__":
    main()