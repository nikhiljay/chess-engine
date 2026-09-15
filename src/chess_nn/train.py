"""Train the eval network on labeled shards, then export and check the ONNX model.

Usage:
  uv run python -m chess_nn.train --name v1-h256 --epochs 30
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from chess_nn.encoding import dense_from_sparse, dense_numpy
from chess_nn.export import export_onnx, parity_check, sanity_checks, validate_with_harness
from chess_nn.harness import RUNS_DIR, SHARDS_DIR
from chess_nn.model import EvalNet, count_params, wdl_loss
from chess_nn.shards import load_shards

PARITY_TOLERANCE = 1e-4


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class TrainConfig:
    name: str
    shards: list[str]
    hidden: int = 256
    epochs: int = 30
    batch_size: int = 16384
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_frac: float = 0.02
    seed: int = 0
    device: str = default_device()
    limit: int | None = None
    runs_dir: str = str(RUNS_DIR)


@torch.no_grad()
def _loss_over(model, features, targets, indices, batch_size) -> float:
    model.eval()
    total = torch.zeros((), device=features.device)
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        out = model(dense_from_sparse(features[batch])).squeeze(1)
        total += wdl_loss(out, targets[batch]) * len(batch)
    return total.item() / len(indices)


def run_training(cfg: TrainConfig) -> Path:
    run_dir = Path(cfg.runs_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    features_np, scores_np = load_shards([Path(p) for p in cfg.shards])
    if cfg.limit:
        features_np, scores_np = features_np[: cfg.limit], scores_np[: cfg.limit]
    n = len(scores_np)

    torch.manual_seed(cfg.seed)
    perm = torch.randperm(n)
    n_val = max(1, int(n * cfg.val_frac))
    val_idx_cpu, train_idx_cpu = perm[:n_val], perm[n_val:]

    device = torch.device(cfg.device)
    features = torch.from_numpy(features_np).to(device)
    targets = torch.from_numpy(scores_np.astype(np.float32)).to(device)
    val_idx, train_idx = val_idx_cpu.to(device), train_idx_cpu.to(device)

    model = EvalNet(cfg.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_idx) / cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * steps_per_epoch)
    print(f"{cfg.name}: {n:,} positions ({len(train_idx):,} train / {n_val:,} val), "
          f"{count_params(model):,} params, device={device}")

    metrics = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start = time.time()
        order = train_idx[torch.randperm(len(train_idx)).to(device)]
        total = torch.zeros((), device=device)
        for s in range(0, len(order), cfg.batch_size):
            batch = order[s : s + cfg.batch_size]
            loss = wdl_loss(model(dense_from_sparse(features[batch])).squeeze(1), targets[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.detach() * len(batch)
        train_loss = total.item() / len(order)
        val_loss = _loss_over(model, features, targets, val_idx, cfg.batch_size)
        metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - start, 1),
        })
        print(f"epoch {epoch:>3}  train {train_loss:.6f}  val {val_loss:.6f}  ({metrics[-1]['seconds']}s)")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        torch.save(model.state_dict(), run_dir / "model.pt")

    model = model.cpu().eval()
    onnx_path = run_dir / "model.onnx"
    export_onnx(model, onnx_path)
    valid = validate_with_harness(onnx_path)
    sample = dense_numpy(features_np[val_idx_cpu[:1000].numpy()])
    parity = parity_check(model, onnx_path, sample)
    sanity = sanity_checks(onnx_path)

    summary = {
        "params": count_params(model),
        "final_train_loss": metrics[-1]["train_loss"],
        "final_val_loss": metrics[-1]["val_loss"],
        "harness_validation": valid,
        "onnx_parity_max_abs_diff": parity,
        "onnx_parity_ok": parity < PARITY_TOLERANCE,
        "sanity": sanity,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--shard-dir", type=Path, default=SHARDS_DIR)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    shards = sorted(str(p) for p in args.shard_dir.glob("shard_*.npz") if ".tmp" not in p.name)
    if not shards:
        raise SystemExit(f"no shards found in {args.shard_dir}")
    cfg = TrainConfig(
        name=args.name,
        shards=shards,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        limit=args.limit,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
