#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_unflatten

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_gpt as pt
import train_gpt_mlx as mlxg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cheap local MLX eval sweep for a PyTorch .pt/.ptz artifact")
    p.add_argument("--artifact", required=True)
    p.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    p.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--model-dim", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--mlp-mult", type=int, default=2)
    p.add_argument("--train-seq-len", type=int, default=1024)
    p.add_argument("--eval-seq-len", type=int, default=1024)
    p.add_argument("--eval-stride", type=int, default=64)
    p.add_argument("--eval-batch-seqs", type=int, default=128)
    p.add_argument("--val-token-limit", type=int, default=1_048_576)
    return p.parse_args()


def load_mlx_state_dict(path: str) -> dict[str, mx.array]:
    state, _, _ = pt.load_init_state_dict(path)
    out: dict[str, mx.array] = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        arr = v.detach().cpu()
        if arr.is_floating_point() and arr.dtype == torch.bfloat16:
            arr = arr.float()
        out[k] = mx.array(arr.numpy())
    return out


def forward_logits(model: mlxg.GPT, x: mx.array) -> mx.array:
    h = model(x)
    logits = h @ model.tok_emb.weight.astype(h.dtype).T
    return model.softcap(logits).astype(mx.float32)


def sliding_windows(total: int, seq_len: int, stride: int) -> list[tuple[int, int, int]]:
    starts = list(range(0, max(total - seq_len, 0) + 1, stride)) or [0]
    last = max(total - seq_len, 0)
    if starts[-1] != last:
        starts.append(last)
    windows: list[tuple[int, int, int]] = []
    next_token = 0
    for start in starts:
        score_from = max(next_token - start, 0)
        score_to = min(total - start, seq_len)
        if score_from < score_to:
            windows.append((start, score_from, score_to))
            next_token = start + score_to
    if next_token != total:
        raise ValueError(f"Sliding coverage mismatch: {next_token}/{total}")
    return windows


def eval_sliding(model: mlxg.GPT, val_tokens: np.ndarray, seq_len: int, stride: int, batch_seqs: int,
                 base_bytes: np.ndarray, lead_space: np.ndarray, boundary: np.ndarray) -> tuple[float, float]:
    windows = sliding_windows(val_tokens.size - 1, seq_len, stride)
    total_loss, total_tokens, total_bytes = 0.0, 0.0, 0.0
    for i in range(0, len(windows), batch_seqs):
        batch = windows[i:i + batch_seqs]
        x_np = np.stack([val_tokens[s:s + seq_len] for s, _, _ in batch]).astype(np.int32, copy=False)
        y_np = np.stack([val_tokens[s + 1:s + seq_len + 1] for s, _, _ in batch]).astype(np.int32, copy=False)
        logits = forward_logits(model, mx.array(x_np, dtype=mx.int32))
        mx.eval(logits)
        for b, (_, score_from, score_to) in enumerate(batch):
            tgt = mx.array(y_np[b, score_from:score_to], dtype=mx.int32)
            loss = nn.losses.cross_entropy(logits[b, score_from:score_to], tgt, reduction="sum")
            mx.eval(loss)
            total_loss += float(loss.item())
            prev = x_np[b, score_from:score_to]
            gold = y_np[b, score_from:score_to]
            bytes_np = base_bytes[gold].astype(np.int16, copy=True)
            bytes_np += (lead_space[gold] & ~boundary[prev]).astype(np.int16, copy=False)
            total_tokens += float(gold.size)
            total_bytes += float(bytes_np.astype(np.float64).sum())
    val_loss = total_loss / total_tokens
    val_bpb = (total_loss / np.log(2.0)) / total_bytes
    return float(val_loss), float(val_bpb)


def main() -> None:
    a = parse_args()
    sp = mlxg.spm.SentencePieceProcessor(model_file=a.tokenizer_path)
    model = mlxg.GPT(
        vocab_size=a.vocab_size, num_layers=a.num_layers, dim=a.model_dim, num_heads=a.num_heads,
        num_kv_heads=a.num_kv_heads, mlp_mult=a.mlp_mult, logit_chunk_tokens=0, logit_softcap=30.0,
        rope_base=10000.0, tied_embed_init_std=0.005, qk_gain_init=1.5,
    )
    model.update(tree_unflatten(list(load_mlx_state_dict(a.artifact).items())))
    flat = dict(tree_flatten(model.parameters()))
    print(f"loaded_params {len(flat)} artifact={a.artifact}")
    val_tokens = mlxg.load_validation_tokens(str(Path(a.data_path) / "fineweb_val_*.bin"), a.eval_seq_len)
    if a.val_token_limit > 0:
        usable = min(val_tokens.size - 1, a.val_token_limit)
        usable = (usable // a.eval_seq_len) * a.eval_seq_len
        val_tokens = val_tokens[:usable + 1]
    base_bytes, lead_space, boundary = mlxg.build_sentencepiece_luts(sp, a.vocab_size)
    val_loss, val_bpb = eval_sliding(
        model, val_tokens, a.eval_seq_len, a.eval_stride, a.eval_batch_seqs,
        base_bytes, lead_space, boundary,
    )
    print(
        f"artifact={a.artifact} eval_seq_len={a.eval_seq_len} eval_stride={a.eval_stride} "
        f"eval_batch_seqs={a.eval_batch_seqs} val_tokens={val_tokens.size - 1} "
        f"val_loss={val_loss:.8f} val_bpb={val_bpb:.8f}"
    )


if __name__ == "__main__":
    main()
