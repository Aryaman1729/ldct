import numpy as np
import torch
from network_transport import DualFiLMSEResUNet, LinkTelemetry, SemanticCTPacketizer
from dataset import window_to_unit

def run_demo():
    # 1. Synthetic 512x512 CT slice (Patient body + surrounding scanner air at -1024 HU)
    Y, X = np.ogrid[:512, :512]
    dist = np.sqrt((X - 256) ** 2 + (Y - 256) ** 2)
    ct_slice = np.full((512, 512), -1024.0, dtype=np.float32)
    ct_slice[dist <= 185] = 45.0   # Soft tissue
    ct_slice[dist <= 60] = 280.0   # Bone / Spine

    # 2. Simulate 5G Mobile CT Van link telemetry (25 Mbps, 35ms RTT, 1.5% packet loss)
    telemetry = LinkTelemetry(bandwidth_mbps=25.0, rtt_ms=35.0, loss_rate=0.015, queue_backlog_slices=1)
    packetizer = SemanticCTPacketizer(target_fps=15.0)

    # 3. Encode into Stream 0 (DSCP EF) + Stream 1 (DSCP AF41) & drop air (DSCP BE)
    pkt = packetizer.encode_slice(ct_slice, dose_frac=0.10, telemetry=telemetry)
    print(f"[CN Layer] Bit-depth: {pkt['bit_depth']}-bit | Link Quality q: {pkt['link_q']} | Bandwidth Saved: {pkt['bandwidth_saving_pct']}%")

    # 4. Receiver Decode + Dual-FiLM-SE-ResUNet Inference
    recon_hu, d_recv, q_recv = packetizer.decode_slice(pkt["stream0_control"], pkt["stream1_roi_data"])
    model = DualFiLMSEResUNet(base=24).eval()
    inp = torch.from_numpy(window_to_unit(recon_hu, -1000.0, 400.0)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        denoised = model(inp, torch.tensor([d_recv]), torch.tensor([q_recv]))

    print(f"[DIP Layer] Output Shape: {tuple(denoised.shape)} | Output Range: [{denoised.min():.3f}, {denoised.max():.3f}]")

if __name__ == "__main__":
    run_demo()