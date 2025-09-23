import torch
from pathlib import Path

# --- make imports work when run directly in PyCharm ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # project root …/ndp_pytorch
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_diffusion_processes.model import BiDimensionalAttentionModel

from config import Config

def print_cross_gates(model):
    import torch
    for li, blk in enumerate(getattr(model, "layers", [])):
        if hasattr(blk, "gamma_d") and hasattr(blk, "gamma_n"):
            gd = float(blk.gamma_d.detach().cpu()); gn = float(blk.gamma_n.detach().cpu())
            gd_eff = float(torch.tanh(blk.gamma_d).detach().cpu())
            gn_eff = float(torch.tanh(blk.gamma_n).detach().cpu())
            print(f"[layer {li:02d}] "
                  f"gamma_d={gd:+.4f}  tanh(gamma_d)={gd_eff:+.4f} | "
                  f"gamma_n={gn:+.4f}  tanh(gamma_n)={gn_eff:+.4f}")

if __name__ == "__main__":
    ckpt = Path("logs/regression/Sep09_205729_tyui/model_ema.pt")  #
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiDimensionalAttentionModel(
        n_layers=cfg.network.n_layers,
        hidden_dim=cfg.network.hidden_dim,
        num_heads=cfg.network.num_heads,
        init_zero=True,
    ).to(device)

    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    print("== Cross-attn gates (raw & tanh) ==")
    print_cross_gates(model)
