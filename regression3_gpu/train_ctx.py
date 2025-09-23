
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import argparse, torch
from neural_diffusion_processes.process import GaussianDiffusion, cosine_schedule
from neural_diffusion_processes.model import ContextConditionedNDP
from regression3_gpu.data_ctx import get_batch_ctx
from regression3_gpu.train import _ema_update, forward_loss_with_context

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dx", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    T = 250
    alphas_cumprod = cosine_schedule(T).to(device)
    process = GaussianDiffusion(alphas_cumprod)

    model = ContextConditionedNDP(d_x=args.dx, d_hidden=args.hidden, n_layers=args.layers, n_heads=args.heads).to(device)
    model_ema = ContextConditionedNDP(d_x=args.dx, d_hidden=args.hidden, n_layers=args.layers, n_heads=args.heads).to(device)
    model_ema.load_state_dict(model.state_dict())
    for p in model_ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)

    for step in range(1, args.steps+1):
        batch = get_batch_ctx(args.batch_size, device=device, dx=args.dx)
        loss = forward_loss_with_context(model, process, batch, device)
        opt.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); _ema_update(model_ema, model, 0.999)
        if step % 50 == 0:
            print(f"[{step}] loss={loss.item():.4f}")
    print("done.")

if __name__ == "__main__":
    main()
