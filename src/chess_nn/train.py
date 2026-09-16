"""Train the eval network on labeled shards, then export and check the ONNX model.

Usage:
  uv run python -m chess_nn.train --name v1-h256 --epochs 30
  uv run python -m chess_nn.train --name h128-s800 --hidden 128 --head none \
      --wdl-scale-cp 800 --shards data/shards/shard_0000.npz
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from chess_nn.encoding import dense_from_sparse, dense_numpy
from chess_nn.export import export_onnx, export_onnx_torch, parity_check, sanity_checks, validate_with_harness
from chess_nn.harness import RUNS_DIR, SHARDS_DIR, VALIDATE_MODEL, load_harness_tool
from chess_nn.groups import load_group_rows, padded
from chess_nn.model import (
    ConvNet,
    EvalNet,
    KingBucketNet,
    ThreatNet,
    count_params,
    cp_loss,
    group_consistency_loss,
    rank_loss,
    wdl_loss,
)
from chess_nn.rank_pairs import KINDS, load_pairs
from chess_nn.shaping import progress_bonus
from chess_nn.shards import load_fens, load_shards

PARITY_TOLERANCE = 1e-4
# Mates further than this many moves share the lowest rung of the mate ladder.
MATE_LADDER_MOVES = 50


def training_targets(
    scores: np.ndarray,
    mates: np.ndarray,
    score_clamp_cp: int,
    mate_base_cp: int | None = None,
    mate_step_cp: int = 20,
    features: np.ndarray | None = None,
    shaping_threshold_cp: int | None = None,
    shaping_pawn_cp: int = 10,
    shaping_trade_cp: int = 4,
    shaping_mopup_cp: int = 15,
) -> np.ndarray:
    """Centipawn targets: clamp scores; optionally rank mates and shape won positions.

    With mate_base_cp set, mate in N becomes ±(mate_base_cp + (50 - N) * mate_step_cp),
    so a shorter mate always scores higher than a longer one. With
    shaping_threshold_cp set, non-mate positions at least that lopsided get a
    progress bonus for the winning side (see chess_nn.shaping).
    """
    targets = np.clip(scores, -score_clamp_cp, score_clamp_cp).astype(np.float32)
    if mate_base_cp is not None:
        is_mate = mates != 0
        distance = np.minimum(np.abs(mates[is_mate].astype(np.int32)), MATE_LADDER_MOVES)
        targets[is_mate] = np.sign(mates[is_mate]) * (mate_base_cp + (MATE_LADDER_MOVES - distance) * mate_step_cp)
    if shaping_threshold_cp is not None:
        won = (np.abs(scores.astype(np.int32)) >= shaping_threshold_cp) & (mates == 0)
        stm_winning = scores[won] > 0
        bonus = progress_bonus(features[won], stm_winning, shaping_pawn_cp, shaping_trade_cp, shaping_mopup_cp)
        targets[won] += np.where(stm_winning, bonus, -bonus)
    return targets


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parse_head(text: str) -> list[int]:
    """'32,32' → [32, 32]; '' or 'none' → []."""
    if text.strip().lower() in ("", "none"):
        return []
    return [int(part) for part in text.split(",")]


@dataclass
class TrainConfig:
    name: str
    shards: list[str]
    hidden: int = 256
    head: list[int] = field(default_factory=lambda: [32, 32])
    epochs: int = 30
    batch_size: int = 16384
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_frac: float = 0.02
    seed: int = 0
    wdl_scale_cp: float = 400.0
    score_clamp_cp: int = 2500
    mate_base_cp: int | None = None
    mate_step_cp: int = 20
    shaping_threshold_cp: int | None = None
    shaping_pawn_cp: int = 10
    shaping_trade_cp: int = 4
    shaping_mopup_cp: int = 15
    # Weight of an auxiliary Huber loss in pawns (0 = WDL loss only). Validation
    # loss stays WDL-only so runs remain comparable.
    cp_loss_weight: float = 0.0
    cp_loss_clip_pawns: float = 50.0
    # "huber" (linear past a pawn) or "mse" (quadratic, so large misses are expensive).
    cp_loss_kind: str = "huber"
    # Feature transformer chosen by each side's own king square (model.KingBucketNet).
    king_buckets: bool = False
    # Width of the attack/threat feature transformer (model.ThreatNet); 0 = off.
    threat_hidden: int = 0
    # Residual CNN over the board (model.ConvNet) instead of the NNUE-style net; 0 = off.
    conv_channels: int = 0
    conv_blocks: int = 6
    # Move-ranking pairs from chess_nn.rank_pairs (paths under training/); rank_weight 0 = off.
    rank_pairs: list[str] = field(default_factory=list)
    rank_val_pairs: list[str] = field(default_factory=list)
    rank_weight: float = 0.0
    rank_margin_pawns: float = 0.2
    rank_temperature_pawns: float = 0.5
    rank_batch_size: int = 4096
    # Pair kinds to train on (rank_pairs.KINDS); validation accuracy always uses all kinds.
    rank_kinds: list[str] = field(default_factory=lambda: list(KINDS))
    # Sibling leaves per decision from chess_nn.decision_data --groups. The loss penalises
    # error that varies inside a group (which changes the move the search plays) and ignores
    # error shared by the whole group (which cancels). group_weight 0 = off.
    group_files: list[str] = field(default_factory=list)
    group_weight: float = 0.0
    group_batch: int = 128
    group_max_members: int = 48
    device: str = field(default_factory=default_device)
    limit: int | None = None
    runs_dir: str = str(RUNS_DIR)


@torch.no_grad()
def _loss_over(model, features, targets, indices, batch_size, scale_cp) -> float:
    model.eval()
    total = torch.zeros((), device=features.device)
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        out = model(dense_from_sparse(features[batch])).squeeze(1)
        total += wdl_loss(out, targets[batch], scale_cp) * len(batch)
    return total.item() / len(indices)


@torch.no_grad()
def _group_error_std(model, features, targets, indices, mask, chunk: int = 256) -> float:
    """Root mean within-group variance of the error, in pawns, over held-out members.

    The part of the error that actually changes which move the search plays.
    """
    if len(indices) == 0:
        return 0.0
    model.eval()
    total = torch.zeros((), device=features.device)
    for start in range(0, len(indices), chunk):
        rows, valid = indices[start : start + chunk], mask[start : start + chunk]
        flat = rows.reshape(-1)
        out = model(dense_from_sparse(features[flat])).squeeze(1).view(rows.shape)
        total += group_consistency_loss(out, targets[flat].view(rows.shape), valid) * len(rows)
    return math.sqrt(total.item() / len(indices))


@torch.no_grad()
def _pair_accuracy(model, pairs, batch_size) -> float:
    """Fraction of pairs where the net scores the better move's child lower (prefers it)."""
    model.eval()
    good, bad = pairs
    correct = 0
    for start in range(0, len(good), batch_size):
        g = model(dense_from_sparse(good[start : start + batch_size])).squeeze(1)
        b = model(dense_from_sparse(bad[start : start + batch_size])).squeeze(1)
        correct += int((g < b).sum().item())
    return correct / max(len(good), 1)


def run_training(cfg: TrainConfig) -> Path:
    run_dir = Path(cfg.runs_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    features_np, scores_np, mates_np = load_shards([Path(p) for p in cfg.shards])
    if cfg.limit:
        features_np, scores_np, mates_np = features_np[: cfg.limit], scores_np[: cfg.limit], mates_np[: cfg.limit]
    n = len(scores_np)

    torch.manual_seed(cfg.seed)
    perm = torch.randperm(n)
    n_val = max(1, int(n * cfg.val_frac))
    val_idx_cpu, train_idx_cpu = perm[:n_val], perm[n_val:]

    device = torch.device(cfg.device)
    features = torch.from_numpy(features_np).to(device)
    targets_np = training_targets(
        scores_np,
        mates_np,
        cfg.score_clamp_cp,
        cfg.mate_base_cp,
        cfg.mate_step_cp,
        features=features_np,
        shaping_threshold_cp=cfg.shaping_threshold_cp,
        shaping_pawn_cp=cfg.shaping_pawn_cp,
        shaping_trade_cp=cfg.shaping_trade_cp,
        shaping_mopup_cp=cfg.shaping_mopup_cp,
    )
    targets = torch.from_numpy(targets_np).to(device)
    val_idx, train_idx = val_idx_cpu.to(device), train_idx_cpu.to(device)
    rank_train = load_pairs(cfg.rank_pairs, device, cfg.rank_kinds) if cfg.rank_weight and cfg.rank_pairs else None
    rank_val = load_pairs(cfg.rank_val_pairs, device) if cfg.rank_val_pairs else None

    group_train = group_val = None
    if cfg.group_files:
        rows_per_decision = load_group_rows(cfg.group_files, load_fens([Path(p) for p in cfg.shards]))
        is_train = np.zeros(n, dtype=bool)
        is_train[train_idx_cpu.numpy()] = True
        train_idx_np, train_mask_np = padded(rows_per_decision, is_train, cfg.group_max_members)
        val_idx_np, val_mask_np = padded(rows_per_decision, ~is_train, cfg.group_max_members)
        group_train = (torch.from_numpy(train_idx_np).to(device), torch.from_numpy(train_mask_np).to(device))
        group_val = (torch.from_numpy(val_idx_np).to(device), torch.from_numpy(val_mask_np).to(device))
        print(f"{len(rows_per_decision):,} decisions matched; {len(train_idx_np):,} train groups "
              f"(≥2 members), {len(val_idx_np):,} held-out groups")

    if cfg.conv_channels:
        model = ConvNet(cfg.conv_channels, cfg.conv_blocks).to(device)
    elif cfg.threat_hidden:
        model = ThreatNet(cfg.hidden, tuple(cfg.head), cfg.threat_hidden).to(device)
    elif cfg.king_buckets:
        model = KingBucketNet(cfg.hidden, tuple(cfg.head)).to(device)
    else:
        model = EvalNet(cfg.hidden, tuple(cfg.head)).to(device)
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
            out = model(dense_from_sparse(features[batch])).squeeze(1)
            loss = wdl_loss(out, targets[batch], cfg.wdl_scale_cp)
            if cfg.cp_loss_weight:
                loss = loss + cfg.cp_loss_weight * cp_loss(out, targets[batch], cfg.cp_loss_clip_pawns, cfg.cp_loss_kind)
            if rank_train is not None:
                pick = torch.randint(0, len(rank_train[0]), (cfg.rank_batch_size,), device=device)
                good = model(dense_from_sparse(rank_train[0][pick])).squeeze(1)
                bad = model(dense_from_sparse(rank_train[1][pick])).squeeze(1)
                loss = loss + cfg.rank_weight * rank_loss(good, bad, cfg.rank_margin_pawns, cfg.rank_temperature_pawns)
            if cfg.group_weight and group_train is not None and len(group_train[0]):
                pick = torch.randint(0, len(group_train[0]), (cfg.group_batch,), device=device)
                rows, valid = group_train[0][pick], group_train[1][pick]
                flat = rows.reshape(-1)
                grouped = model(dense_from_sparse(features[flat])).squeeze(1).view(rows.shape)
                loss = loss + cfg.group_weight * group_consistency_loss(
                    grouped, targets[flat].view(rows.shape), valid
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.detach() * len(batch)
        train_loss = total.item() / len(order)
        val_loss = _loss_over(model, features, targets, val_idx, cfg.batch_size, cfg.wdl_scale_cp)
        metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "rank_val_accuracy": _pair_accuracy(model, rank_val, cfg.batch_size) if rank_val is not None else None,
            "group_error_std": _group_error_std(model, features, targets, *group_val) if group_val is not None else None,
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - start, 1),
        })
        print(f"epoch {epoch:>3}  train {train_loss:.6f}  val {val_loss:.6f}  ({metrics[-1]['seconds']}s)")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        torch.save(model.state_dict(), run_dir / "model.pt")

    model = model.cpu().eval()
    onnx_path = run_dir / "model.onnx"
    (export_onnx_torch if isinstance(model, ConvNet) else export_onnx)(model, onnx_path)
    valid = validate_with_harness(onnx_path)
    sample = dense_numpy(features_np[val_idx_cpu[:1000].numpy()])
    parity = parity_check(model, onnx_path, sample)
    sanity = sanity_checks(onnx_path)

    summary = {
        "params": load_harness_tool(VALIDATE_MODEL, "validate_model").count_onnx_params(str(onnx_path)),
        "positions": n,
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
    parser.add_argument("--shards", type=Path, nargs="+", help="explicit shard files (overrides --shard-dir)")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--head", default="32,32", help="hidden widths after the accumulator, or 'none'")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wdl-scale-cp", type=float, default=400.0)
    parser.add_argument("--score-clamp-cp", type=int, default=2500)
    parser.add_argument("--mate-base-cp", type=int, help="rank mates above the clamp starting here (default: off)")
    parser.add_argument("--mate-step-cp", type=int, default=20)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.shards:
        shards = [str(p) for p in args.shards]
    else:
        shards = sorted(str(p) for p in args.shard_dir.glob("shard_*.npz") if ".tmp" not in p.name)
    if not shards:
        raise SystemExit(f"no shards found in {args.shard_dir}")
    cfg = TrainConfig(
        name=args.name,
        shards=shards,
        hidden=args.hidden,
        head=parse_head(args.head),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        wdl_scale_cp=args.wdl_scale_cp,
        score_clamp_cp=args.score_clamp_cp,
        mate_base_cp=args.mate_base_cp,
        mate_step_cp=args.mate_step_cp,
        device=args.device,
        limit=args.limit,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
