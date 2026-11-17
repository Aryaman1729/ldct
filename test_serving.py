"""test_serving.py - end-to-end test of the client-server setup.

Starts a real server (uvicorn) in a background process, then checks the API through client.py:
connection, authentication, model listing, enhancement (with metrics), and the error cases.

    python test_serving.py

Needs at least one trained model (runs/<organ>/<arch>/best.pt) and the cached test data of that organ.
Takes about a minute.  Prints PASS / FAIL for each check and exits with code 1 if anything failed.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from client import LDCTClient, LDCTError  # noqa: E402

KEY = "test-key-123"
failures: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f"  ({extra})" if extra else ""))
    if not ok:
        failures.append(name)


def expect_error(name: str, fn, contains: str) -> None:
    try:
        fn()
    except LDCTError as exc:
        check(name, contains in str(exc), str(exc)[:90])
    else:
        check(name, False, "no error was raised")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    port = free_port()
    env = dict(os.environ, LDCT_API_KEY=KEY, LDCT_MAX_UPLOAD_MB="8")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port),
                               "--log-level", "warning"], cwd=HERE, env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):  # wait up to ~2 minutes (models are loaded at start-up)
            try:
                if requests.get(base + "/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if server.poll() is not None:
                raise SystemExit("the server process died during start-up (see the messages above)")
            time.sleep(1)
        else:
            raise SystemExit("the server did not start within 2 minutes")

        anon, c = LDCTClient(base), LDCTClient(base, KEY)

        h = anon.health()
        check("health works without a key", h["status"] == "ok" and h["auth_required"] is True, f"device {h['device']}")
        expect_error("models rejects a missing key", anon.models, "401")
        expect_error("models rejects a wrong key", lambda: LDCTClient(base, "nope").models(), "401")

        models = c.models()
        check("models lists at least one trained model", len(models) >= 1, ", ".join(f"{m['organ']}/{m['arch']}" for m in models))
        if not models:
            raise SystemExit("no trained model found - train one first (python model.py train --organ liver)")
        m0 = next((m for m in models if m["default"]), models[0])
        organ, arch = m0["organ"], m0["arch"]

        from dataset import cache_dir_for, get_splits

        cache = cache_dir_for(organ)
        key = get_splits(cache)["test"][0]
        low = np.load(cache / key / "low.npy").astype(np.float32)
        full = np.load(cache / key / "full.npy").astype(np.float32)
        z = low.shape[0] // 2

        r = c.enhance(low, organ, arch, z=z, reference=full[z])
        check("enhance returns an HU array of the right shape", r.enhanced is not None and r.enhanced.shape == low.shape[1:])
        check("enhanced values are finite", bool(np.isfinite(r.enhanced).all()))
        check("PNG outputs are real PNG files", r.png[:4] == b"\x89PNG" and r.input_png[:4] == b"\x89PNG")
        check("metrics are returned when a reference is sent", r.metrics is not None and "psnr_out" in r.metrics)
        if r.metrics:
            gain = r.metrics["psnr_out"] - r.metrics["psnr_in"]
            print(f"      PSNR {r.metrics['psnr_in']:.2f} -> {r.metrics['psnr_out']:.2f} dB  (gain {gain:+.2f}),"
                  f" {r.meta['inference_ms']} ms on the server")

        from model import enhance_slice, load_model, pick_device, run_dir_for

        dev = pick_device("cpu")
        local_model, local_ck = load_model(run_dir_for(organ, arch) / "best.pt", dev)
        local = enhance_slice(local_model, local_ck, low, z, dev)
        diff = float(np.abs(local - r.enhanced).max())
        check("server output equals running the model locally", diff < 1.0, f"max difference {diff:.4f} HU")

        r2 = c.enhance(low, organ, None, z=z)  # arch omitted -> server picks the organ's default model
        check("default model is used when arch is omitted", r2.meta["model"]["arch"] == arch and r2.metrics is None)
        r3 = c.enhance(low[z], organ, arch)  # a single 2-D slice, no neighbours
        check("a single 2-D slice works", r3.enhanced.shape == low.shape[1:] and r3.meta["n_slices"] == 1)

        expect_error("unknown organ is rejected", lambda: c.enhance(low[z], "brain"), "unknown organ")
        expect_error("untrained architecture is rejected", lambda: c.enhance(low[z], organ, "restormer"), "no trained model")
        expect_error("out-of-range z is rejected", lambda: c.enhance(low, organ, arch, z=10_000), "outside")
        expect_error("NaN input is rejected", lambda: c.enhance(np.full((128, 128), np.nan), organ, arch), "NaN")
        expect_error("tiny image is rejected", lambda: c.enhance(np.zeros((8, 8)), organ, arch), "outside the accepted range")
        expect_error("oversized upload is rejected", lambda: c.enhance(np.zeros((3, 1024, 1024), np.float32), organ, arch, context=5), "413")
        junk = HERE / "_junk.txt"
        junk.write_text("not an image")
        try:
            expect_error("wrong file type is rejected", lambda: c.enhance(junk, organ, arch), "must be .dcm")
        finally:
            junk.unlink(missing_ok=True)
        expect_error("unreachable server gives a clear message", lambda: LDCTClient("http://127.0.0.1:1").health(), "cannot reach")
    finally:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
