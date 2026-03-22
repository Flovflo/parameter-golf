#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_gpt_mlx as t


def make_batches(tokens: np.ndarray, seq_len: int, batch_tokens: int, limit: int) -> list[tuple[mx.array, mx.array]]:
    usable = (batch_tokens // seq_len) * seq_len
    chunk = usable + 1
    batches = []
    for i in range(0, len(tokens) - chunk + 1, usable):
        sl = tokens[i : i + chunk]
        x = mx.array(sl[:-1].reshape(-1, seq_len), dtype=mx.int32)
        y = mx.array(sl[1:].reshape(-1, seq_len), dtype=mx.int32)
        batches.append((x, y))
        if len(batches) >= limit:
            break
    return batches


def build_args(**kwargs) -> t.Hyperparameters:
    args = t.Hyperparameters()
    args.vocab_size = 1024
    args.num_layers = 10
    args.model_dim = 512
    args.num_heads = 8
    args.num_kv_heads = 4
    args.mlp_mult = 2
    args.mlp_hidden = 1536
    args.tied_embed_lr = 0.05
    args.matrix_lr = 0.04
    args.scalar_lr = 0.04
    args.beta1 = 0.9
    args.beta2 = 0.95
    args.adam_eps = 1e-8
    args.muon_momentum = 0.95
    args.muon_backend_steps = 5
    args.muon_momentum_warmup_start = 0.85
    args.muon_momentum_warmup_steps = 500
    args.logit_chunk_tokens = 0
    args.logit_softcap = 30.0
    args.rope_base = 10000.0
    args.tied_embed_init_std = 0.005
    args.qk_gain_init = 1.5
    args.bigram_hash_buckets = 0
    args.trigram_hash_buckets = 0
    args.hash_dim = 64
    args.hash_gate = True
    args.hash_mixer_hidden = 0
    args.hash_memory_buckets = 0
    args.hash_memory_dim = 64
    args.hash_memory_recent_weight = 0.2
    args.hash_memory_gate = True
    args.hash_memory_chunk_size = 1
    args.hash_memory_min_count = 0
    args.hash_memory_mode = "input"
    args.hash_memory_count_alpha = 0.0
    args.hash_memory_scale = 1.0
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def build_model(args: t.Hyperparameters) -> t.GPT:
    return t.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        bigram_hash_buckets=args.bigram_hash_buckets,
        trigram_hash_buckets=args.trigram_hash_buckets,
        hash_dim=args.hash_dim,
        hash_gate=args.hash_gate,
        hash_mixer_hidden=args.hash_mixer_hidden,
        hash_memory_buckets=args.hash_memory_buckets,
        hash_memory_dim=args.hash_memory_dim,
        hash_memory_recent_weight=args.hash_memory_recent_weight,
        hash_memory_gate=args.hash_memory_gate,
        hash_memory_chunk_size=args.hash_memory_chunk_size,
        hash_memory_min_count=args.hash_memory_min_count,
        hash_memory_mode=args.hash_memory_mode,
        hash_memory_count_alpha=args.hash_memory_count_alpha,
        hash_memory_scale=args.hash_memory_scale,
    )


def run_trial(name: str, args: t.Hyperparameters, train_batches, val_batch) -> dict[str, float | int | str]:
    mx.random.seed(1337)
    model = build_model(args)
    opt = t.SplitOptimizers(model, args)
    use_compile = args.hash_memory_buckets <= 0
    loss_fn = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state) if use_compile else lambda x, y: model.loss(x, y)
    loss_and_grad = (
        mx.compile(nn.value_and_grad(model, lambda x, y: model.loss(x, y)), inputs=model.state, outputs=model.state)
        if use_compile
        else nn.value_and_grad(model, lambda x, y: model.loss(x, y))
    )
    x_val, y_val = val_batch
    start_val = float(loss_fn(x_val, y_val).item())
    train_losses = []
    for step, (x, y) in enumerate(train_batches, start=1):
        loss, grads = loss_and_grad(x, y)
        mx.eval(loss, grads)
        train_losses.append(float(loss.item()))
        opt.step(model, grads, step=step, lr_mul=1.0)
        mx.eval(model.state)
    end_val = float(loss_fn(x_val, y_val).item())
    params = sum(int(np.prod(p.shape)) for _, p in t.tree_flatten(model.parameters()))
    return {
        "name": name,
        "params": params,
        "start_val_loss": start_val,
        "end_val_loss": end_val,
        "delta_val_loss": end_val - start_val,
        "last_train_loss": train_losses[-1],
        "used_compile": int(use_compile),
    }


def main() -> None:
    train_tokens = t.load_data_shard(Path("data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin"))
    val_tokens = t.load_data_shard(Path("data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"))
    train_batches = make_batches(train_tokens[: (12 * 4096 + 1)], seq_len=1024, batch_tokens=4096, limit=12)
    val_batch = make_batches(val_tokens[: (8 * 1024 + 1)], seq_len=1024, batch_tokens=8192, limit=1)[0]
    trials = [
        ("baseline", build_args()),
        ("bigram16384", build_args(bigram_hash_buckets=16384)),
        ("dynmem16384_c16", build_args(hash_memory_buckets=16384, hash_memory_chunk_size=16)),
        ("dynmem16384_c32_m1", build_args(hash_memory_buckets=16384, hash_memory_chunk_size=32, hash_memory_min_count=1)),
        ("dynmem16384_c32_m2", build_args(hash_memory_buckets=16384, hash_memory_chunk_size=32, hash_memory_min_count=2)),
    ]
    results = [run_trial(name, args, train_batches, val_batch) for name, args in trials]
    out = Path("logs/mlx_hash_dynamic_loop.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
