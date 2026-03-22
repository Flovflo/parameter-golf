#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_unflatten

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_gpt as pt
import train_gpt_mlx as mlxg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLX doc-local prefix adaptation eval")
    p.add_argument("--artifact", required=True)
    p.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    p.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--model-dim", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--mlp-mult", type=int, default=2)
    p.add_argument("--mlp-hidden", type=int, default=0)
    p.add_argument("--eval-seq-len", type=int, default=1024)
    p.add_argument("--eval-stride", type=int, default=64)
    p.add_argument("--eval-batch-seqs", type=int, default=16)
    p.add_argument(
        "--compiled-eval-mode",
        choices=("none", "prefix_gates", "ngram_cache", "prefix_gates_ngram_cache"),
        default="none",
    )
    p.add_argument("--prefix-tokens", type=int, default=256)
    p.add_argument("--doc-limit", type=int, default=64)
    p.add_argument("--adapt-layers", type=int, default=2)
    p.add_argument("--gate-cap", type=float, default=0.08)
    p.add_argument("--gate-targets", default="skip_weights,attn_scale,mlp_scale")
    p.add_argument("--cache-ngrams", default="8,4,2")
    p.add_argument("--cache-backend", choices=("exact", "bigram_hash", "multi_hash"), default="exact")
    p.add_argument("--hash-buckets", type=int, default=10240)
    p.add_argument("--cache-alpha", type=float, default=0.35)
    p.add_argument("--cache-alpha-hits", type=float, default=1.0)
    p.add_argument("--cache-min-count", type=int, default=1)
    p.add_argument("--cache-update-suffix", type=int, default=1)
    p.add_argument("--cache-recent-weight", type=float, default=0.0)
    p.add_argument("--smear-gate", type=int, default=0)
    p.add_argument("--smear-entropy-low", type=float, default=1.5)
    p.add_argument("--smear-entropy-high", type=float, default=4.0)
    return p.parse_args()


def load_model(a: argparse.Namespace) -> tuple[mlxg.GPT, mlxg.spm.SentencePieceProcessor]:
    state, _, _ = pt.load_init_state_dict(a.artifact)
    flat = {k: mx.array(v.detach().cpu().float().numpy()) for k, v in state.items() if isinstance(v, torch.Tensor)}
    model = mlxg.GPT(
        vocab_size=a.vocab_size, num_layers=a.num_layers, dim=a.model_dim, num_heads=a.num_heads,
        num_kv_heads=a.num_kv_heads, mlp_mult=a.mlp_mult, mlp_hidden=a.mlp_hidden,
        logit_chunk_tokens=0, logit_softcap=30.0, rope_base=10000.0, tied_embed_init_std=0.005, qk_gain_init=1.5,
    )
    model.update(tree_unflatten(list(flat.items())))
    return model, mlxg.spm.SentencePieceProcessor(model_file=a.tokenizer_path)


def block_forward(block, x: mx.array, x0: mx.array, ctl: dict | None) -> mx.array:
    mix = block.resid_mix.astype(x.dtype)
    x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    attn_scale = block.attn_scale.astype(x.dtype) * float((ctl or {}).get("attn", 1.0))
    mlp_scale = block.mlp_scale.astype(x.dtype) * float((ctl or {}).get("mlp", 1.0))
    x = x + attn_scale[None, None, :] * block.attn(block.attn_norm(x))
    x = x + mlp_scale[None, None, :] * block.mlp(block.mlp_norm(x))
    return x


def forward_hidden(model: mlxg.GPT, ids: mx.array, adapt: dict | None = None) -> mx.array:
    x = mlxg.rms_norm(model.tok_emb(ids).astype(mlxg.COMPUTE_DTYPE))
    x0, blocks, skip_ctl = x, (adapt or {}).get("blocks", {}), (adapt or {}).get("skip", {})
    skips: list[mx.array] = []
    for i in range(model.num_encoder_layers):
        x = block_forward(model.blocks[i], x, x0, blocks.get(i))
        skips.append(x)
    for i in range(model.num_decoder_layers):
        if skips:
            gate = float(skip_ctl.get(i, 1.0))
            x = x + gate * model.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
        bi = model.num_encoder_layers + i
        x = block_forward(model.blocks[bi], x, x0, blocks.get(bi))
    return model.final_norm(x)


def forward_logits(model: mlxg.GPT, ids: mx.array, adapt: dict | None = None) -> mx.array:
    h = forward_hidden(model, ids, adapt)
    logits = h @ model.tok_emb.weight.astype(h.dtype).T
    return model.softcap(logits).astype(mx.float32)


def find_docs(tokens: np.ndarray, bos_id: int, limit: int) -> list[tuple[int, int]]:
    pos = np.flatnonzero(tokens == bos_id)
    out: list[tuple[int, int]] = []
    for i, start in enumerate(pos):
        end = int(pos[i + 1]) if i + 1 < pos.size else int(tokens.size)
        if end - int(start) > 2:
            out.append((int(start), end - int(start)))
        if limit > 0 and len(out) >= limit:
            break
    return out


def sliding_windows(total: int, seq_len: int, stride: int, score_start: int) -> list[tuple[int, int, int]]:
    starts = list(range(0, max(total - seq_len, 0) + 1, stride)) or [0]
    last = max(total - seq_len, 0)
    if starts[-1] != last:
        starts.append(last)
    wins, next_token = [], score_start
    for start in starts:
        score_from, score_to = max(next_token - start, 0), min(total - start, seq_len)
        if score_from < score_to:
            wins.append((start, score_from, score_to))
            next_token = start + score_to
    if next_token != total:
        raise ValueError(f"Sliding coverage mismatch: {next_token}/{total}")
    return wins


def cosine_gate(latent: mx.array, vec: mx.array, cap: float) -> float:
    lat = latent.astype(mx.float32)
    lat = lat / (mx.sqrt(mx.mean(lat * lat)) + 1e-6)
    ref = vec.astype(mx.float32)
    ref = ref / (mx.sqrt(mx.mean(ref * ref)) + 1e-6)
    return float(1.0 + cap * np.tanh(float(mx.mean(lat * ref).item())))


def build_adapt(model: mlxg.GPT, latent: mx.array, layers: int, cap: float, targets: set[str]) -> dict:
    adapt = {"blocks": {}, "skip": {}}
    for bi in range(max(0, len(model.blocks) - layers), len(model.blocks)):
        ctl = {}
        if "attn_scale" in targets:
            ctl["attn"] = cosine_gate(latent, model.blocks[bi].attn_scale, cap)
        if "mlp_scale" in targets:
            ctl["mlp"] = cosine_gate(latent, model.blocks[bi].mlp_scale, cap)
        if ctl:
            adapt["blocks"][bi] = ctl
    if "skip_weights" in targets:
        for si in range(max(0, model.num_skip_weights - layers), model.num_skip_weights):
            adapt["skip"][si] = cosine_gate(latent, model.skip_weights[si], cap)
    return adapt


def parse_ngram_list(spec: str) -> tuple[int, ...]:
    vals = sorted({int(x.strip()) for x in spec.split(",") if x.strip()}, reverse=True)
    if any(v <= 0 for v in vals):
        raise ValueError(f"cache n-grams must be positive, got {spec!r}")
    return tuple(vals)


def build_ngram_cache(raw: np.ndarray, prefix_len: int, ngrams: tuple[int, ...]) -> dict[int, dict[tuple[int, ...], Counter]]:
    caches: dict[int, dict[tuple[int, ...], Counter]] = {n: defaultdict(Counter) for n in ngrams}
    for target_idx in range(1, prefix_len):
        update_ngram_cache(caches, raw, target_idx, ngrams)
    return caches


def update_ngram_cache(
    caches: dict[int, dict[tuple[int, ...], Counter]],
    raw: np.ndarray,
    target_idx: int,
    ngrams: tuple[int, ...],
) -> None:
    tgt = int(raw[target_idx])
    for n in ngrams:
        if target_idx < n:
            continue
        ctx = tuple(int(x) for x in raw[target_idx - n:target_idx])
        caches[n][ctx][tgt] += 1


def query_ngram_cache(
    caches: dict[int, dict[tuple[int, ...], Counter]],
    raw: np.ndarray,
    prev_idx: int,
    target: int,
    ngrams: tuple[int, ...],
    min_count: int,
) -> tuple[float | None, int, int, float]:
    for n in ngrams:
        if prev_idx + 1 < n:
            continue
        ctx = tuple(int(x) for x in raw[prev_idx - n + 1:prev_idx + 1])
        counts = caches[n].get(ctx)
        if not counts:
            continue
        total = sum(counts.values())
        if total < min_count:
            continue
        peak = max(counts.values()) / total
        return counts.get(int(target), 0) / total, n, total, peak
    return None, 0, 0, 0.0


def hash_bigram(t0: int, t1: int, buckets: int) -> int:
    return int((((t0 + 1) * 1315423911) ^ ((t1 + 1) * 2654435761)) % buckets)


def hash_ngram(ctx: np.ndarray, buckets: int) -> int:
    h = 1469598103934665603
    for tok in ctx:
        h ^= int(tok) + 1
        h *= 1099511628211
    return int(h % buckets)


def build_bigram_hash(
    raw: np.ndarray, prefix_len: int, buckets: int, vocab: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros((buckets, vocab), dtype=np.uint16)
    totals = np.zeros((buckets,), dtype=np.uint32)
    recent = np.full((buckets,), -1, dtype=np.int32)
    for target_idx in range(2, prefix_len):
        update_bigram_hash(counts, totals, recent, raw, target_idx)
    return counts, totals, recent


def update_bigram_hash(
    counts: np.ndarray, totals: np.ndarray, recent: np.ndarray, raw: np.ndarray, target_idx: int
) -> None:
    if target_idx < 2:
        return
    bucket = hash_bigram(int(raw[target_idx - 2]), int(raw[target_idx - 1]), int(counts.shape[0]))
    tgt = int(raw[target_idx])
    if counts[bucket, tgt] < np.iinfo(np.uint16).max:
        counts[bucket, tgt] += 1
    totals[bucket] += 1
    recent[bucket] = tgt


def query_bigram_hash(
    counts: np.ndarray,
    totals: np.ndarray,
    recent: np.ndarray,
    raw: np.ndarray,
    prev_idx: int,
    target: int,
    min_count: int,
) -> tuple[float | None, int, int, float, int]:
    if prev_idx < 1:
        return None, 0, 0, 0.0, -1
    bucket = hash_bigram(int(raw[prev_idx - 1]), int(raw[prev_idx]), int(counts.shape[0]))
    total = int(totals[bucket])
    if total < min_count:
        return None, 0, 0, 0.0, int(recent[bucket])
    row = counts[bucket]
    peak = float(int(row.max())) / total
    return float(int(row[int(target)])) / total, 2, total, peak, int(recent[bucket])


def build_multi_hash(raw: np.ndarray, prefix_len: int, buckets: int, vocab: int, ngrams: tuple[int, ...]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    tables = {
        n: (np.zeros((buckets, vocab), dtype=np.uint16), np.zeros((buckets,), dtype=np.uint32))
        for n in ngrams
    }
    for target_idx in range(1, prefix_len):
        update_multi_hash(tables, raw, target_idx, ngrams)
    return tables


def update_multi_hash(
    tables: dict[int, tuple[np.ndarray, np.ndarray]],
    raw: np.ndarray,
    target_idx: int,
    ngrams: tuple[int, ...],
) -> None:
    tgt = int(raw[target_idx])
    for n in ngrams:
        if target_idx < n:
            continue
        counts, totals = tables[n]
        bucket = hash_ngram(raw[target_idx - n:target_idx], int(counts.shape[0]))
        if counts[bucket, tgt] < np.iinfo(np.uint16).max:
            counts[bucket, tgt] += 1
        totals[bucket] += 1


def query_multi_hash(
    tables: dict[int, tuple[np.ndarray, np.ndarray]],
    raw: np.ndarray,
    prev_idx: int,
    target: int,
    ngrams: tuple[int, ...],
    min_count: int,
) -> tuple[float | None, int, int, float]:
    for n in ngrams:
        if prev_idx + 1 < n:
            continue
        counts, totals = tables[n]
        bucket = hash_ngram(raw[prev_idx - n + 1:prev_idx + 1], int(counts.shape[0]))
        total = int(totals[bucket])
        if total < min_count:
            continue
        row = counts[bucket]
        peak = float(int(row.max())) / total
        return float(int(row[int(target)])) / total, n, total, peak
    return None, 0, 0, 0.0


def eval_docs(model: mlxg.GPT, sp, a: argparse.Namespace) -> tuple[float, float, float, int, int]:
    toks = mlxg.load_validation_tokens_raw(str(Path(a.data_path) / "fineweb_val_*.bin"))
    docs = find_docs(toks, int(sp.bos_id()), a.doc_limit)
    base, lead, boundary = mlxg.build_sentencepiece_luts(sp, a.vocab_size)
    enable_gates = a.compiled_eval_mode in {"prefix_gates", "prefix_gates_ngram_cache"}
    enable_cache = a.compiled_eval_mode in {"ngram_cache", "prefix_gates_ngram_cache"}
    ngrams = parse_ngram_list(a.cache_ngrams) if enable_cache else ()
    targets, loss_sum, token_sum, byte_sum, adapt_ms = set(a.gate_targets.split(",")), 0.0, 0, 0.0, 0.0
    for ds, dl in docs:
        prefix_len = min(a.prefix_tokens, dl - 1)
        if dl - 1 <= prefix_len - 1:
            continue
        raw = toks[ds:ds + dl]
        adapt = None
        if enable_gates:
            t0 = time.perf_counter()
            latent = mx.mean(forward_hidden(model, mx.array(toks[ds:ds + prefix_len][None, :], dtype=mx.int32))[0], axis=0)
            mx.eval(latent)
            adapt = build_adapt(model, latent, a.adapt_layers, a.gate_cap, targets)
            adapt_ms += 1000.0 * (time.perf_counter() - t0)
        hash_cache = None
        multi_hash = None
        cache = None
        if enable_cache:
            if a.cache_backend == "exact":
                cache = build_ngram_cache(raw, prefix_len, ngrams)
            elif a.cache_backend == "bigram_hash":
                hash_cache = build_bigram_hash(raw, prefix_len, a.hash_buckets, a.vocab_size)
            else:
                multi_hash = build_multi_hash(raw, prefix_len, a.hash_buckets, a.vocab_size, ngrams)
        wins = sliding_windows(dl - 1, a.eval_seq_len, a.eval_stride, prefix_len - 1)
        for i in range(0, len(wins), a.eval_batch_seqs):
            batch = wins[i:i + a.eval_batch_seqs]
            x_np = np.stack([raw[s:s + a.eval_seq_len] for s, _, _ in batch]).astype(np.int32, copy=False)
            y_np = np.stack([raw[s + 1:s + a.eval_seq_len + 1] for s, _, _ in batch]).astype(np.int32, copy=False)
            logits = forward_logits(model, mx.array(x_np, dtype=mx.int32), adapt).astype(mx.float32)
            mx.eval(logits)
            if enable_cache:
                log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                mx.eval(log_probs)
                log_probs_np = np.array(log_probs)
                for b, (start, score_from, score_to) in enumerate(batch):
                    for pos in range(score_from, score_to):
                        target = int(y_np[b, pos])
                        prev = int(x_np[b, pos])
                        target_idx = start + pos + 1
                        model_prob = math.exp(float(log_probs_np[b, pos, target]))
                        final_prob = model_prob
                        if a.cache_backend == "exact":
                            cache_prob, _, total, peak = query_ngram_cache(
                                cache, raw, target_idx - 1, target, ngrams, a.cache_min_count
                            )
                        elif a.cache_backend == "bigram_hash":
                            cache_prob, _, total, peak, recent_tgt = query_bigram_hash(
                                hash_cache[0], hash_cache[1], hash_cache[2], raw, target_idx - 1, target, a.cache_min_count
                            )
                        else:
                            cache_prob, _, total, peak = query_multi_hash(
                                multi_hash, raw, target_idx - 1, target, ngrams, a.cache_min_count
                            )
                        if cache_prob is not None:
                            mix = a.cache_alpha
                            if a.cache_alpha_hits > 0:
                                mix *= total / (total + a.cache_alpha_hits)
                            if a.smear_gate:
                                lp = log_probs_np[b, pos]
                                probs = np.exp(lp.astype(np.float64))
                                entropy = float(-(probs * lp).sum())
                                span = max(a.smear_entropy_high - a.smear_entropy_low, 1e-6)
                                ent_gate = min(max((entropy - a.smear_entropy_low) / span, 0.0), 1.0)
                                mix *= peak * ent_gate
                            if a.cache_backend == "bigram_hash" and a.cache_recent_weight > 0.0 and recent_tgt >= 0:
                                cache_prob = (1.0 - a.cache_recent_weight) * cache_prob + a.cache_recent_weight * float(recent_tgt == target)
                            final_prob = (1.0 - mix) * model_prob + mix * cache_prob
                        loss_sum += -math.log(max(final_prob, 1e-12))
                        token_sum += 1
                        byte_sum += float(base[target] + (lead[target] & ~boundary[prev]))
                        if a.cache_update_suffix:
                            if a.cache_backend == "exact":
                                update_ngram_cache(cache, raw, target_idx, ngrams)
                            elif a.cache_backend == "bigram_hash":
                                update_bigram_hash(hash_cache[0], hash_cache[1], hash_cache[2], raw, target_idx)
                            else:
                                update_multi_hash(multi_hash, raw, target_idx, ngrams)
            else:
                for b, (_, score_from, score_to) in enumerate(batch):
                    tgt = mx.array(y_np[b, score_from:score_to], dtype=mx.int32)
                    loss = nn.losses.cross_entropy(logits[b, score_from:score_to], tgt, reduction="sum")
                    mx.eval(loss)
                    gold, prev = y_np[b, score_from:score_to], x_np[b, score_from:score_to]
                    bytes_np = base[gold].astype(np.int16, copy=True) + (lead[gold] & ~boundary[prev]).astype(np.int16, copy=False)
                    loss_sum += float(loss.item())
                    token_sum += int(gold.size)
                    byte_sum += float(bytes_np.astype(np.float64).sum())
    return loss_sum / token_sum, (loss_sum / np.log(2.0)) / byte_sum, adapt_ms, len(docs), token_sum


def main() -> None:
    a = parse_args()
    model, sp = load_model(a)
    t0 = time.perf_counter()
    val_loss, val_bpb, adapt_ms, docs, tokens = eval_docs(model, sp, a)
    total_ms = 1000.0 * (time.perf_counter() - t0)
    print(
        f"artifact={a.artifact} mode={a.compiled_eval_mode} prefix_tokens={a.prefix_tokens} "
        f"adapt_layers={a.adapt_layers} gate_targets={a.gate_targets} gate_cap={a.gate_cap:.4f} "
        f"cache_backend={a.cache_backend} hash_buckets={a.hash_buckets} smear_gate={a.smear_gate} "
        f"cache_recent_weight={a.cache_recent_weight} "
        f"docs={docs} scored_tokens={tokens} val_loss={val_loss:.8f} val_bpb={val_bpb:.8f} "
        f"adapt_time_ms={adapt_ms:.1f} total_eval_ms={total_ms:.1f}"
    )


if __name__ == "__main__":
    main()
