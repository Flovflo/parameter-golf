#!/usr/bin/env python3
"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# ==============================================================================
# SHARD FORMAT + COMPUTE DTYPE
# ==============================================================================

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap
class Hyperparameters:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop. These defaults now mirror train_gpt.py on a single process.
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    # Validation always uses the full fineweb_val split.
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    require_data_manifest: bool = bool(int(os.environ.get("REQUIRE_DATA_MANIFEST", "1")))
    require_full_val_split: bool = bool(int(os.environ.get("REQUIRE_FULL_VAL_SPLIT", "1")))
    # Chunk each logical MLX microbatch into smaller sub-batches to reduce peak
    # memory pressure without changing the effective optimizer batch.
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    # Force MLX to materialize the graph after every sub-batch, preventing lazy
    # graph buildup across accumulation steps. Keeps peak memory low on 16GB machines.
    # Disable on 32GB+ unified memory for better throughput (MLX_EAGER_EVAL=0).
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model (defaults match the current baseline setup).
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    mlp_hidden: int = int(os.environ.get("MLP_HIDDEN", 0))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims: int = int(os.environ.get("ROPE_DIMS", 0))  # 0 = full head_dim; >0 = partial RoPE
    ln_scale: bool = bool(int(os.environ.get("LN_SCALE", "0")))  # scale norms by 1/sqrt(layer+1)
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))
    bigram_hash_buckets: int = int(os.environ.get("BIGRAM_HASH_BUCKETS", 0))
    trigram_hash_buckets: int = int(os.environ.get("TRIGRAM_HASH_BUCKETS", 0))
    hash_dim: int = int(os.environ.get("HASH_DIM", 64))
    hash_gate: bool = bool(int(os.environ.get("HASH_GATE", "1")))
    hash_mixer_hidden: int = int(os.environ.get("HASH_MIXER_HIDDEN", 0))
    hash_memory_buckets: int = int(os.environ.get("HASH_MEMORY_BUCKETS", 0))
    hash_memory_dim: int = int(os.environ.get("HASH_MEMORY_DIM", 64))
    hash_memory_recent_weight: float = float(os.environ.get("HASH_MEMORY_RECENT_WEIGHT", 0.2))
    hash_memory_gate: bool = bool(int(os.environ.get("HASH_MEMORY_GATE", "1")))
    hash_memory_chunk_size: int = int(os.environ.get("HASH_MEMORY_CHUNK_SIZE", 1))
    hash_memory_min_count: int = int(os.environ.get("HASH_MEMORY_MIN_COUNT", 0))
    hash_memory_mode: str = os.environ.get("HASH_MEMORY_MODE", "input")
    hash_memory_count_alpha: float = float(os.environ.get("HASH_MEMORY_COUNT_ALPHA", 0.0))
    hash_memory_scale: float = float(os.environ.get("HASH_MEMORY_SCALE", 1.0))
    smear_gate: bool = bool(int(os.environ.get("SMEAR_GATE", "0")))
    simple_bigram: bool = bool(int(os.environ.get("SIMPLE_BIGRAM", "0")))

    # Optimizer. We keep the same per-group defaults as train_gpt.py.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP16_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("INT8_KEEP_FLOAT_FP16_NAME_PATTERNS", "tok_emb.weight").split(",")
    if pattern
)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    # Background on Muon: https://kellerjordan.github.io/posts/muon/
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    # MLX module wrapper around the functional RMSNorm helper so it composes nicely in blocks.
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    # - separate q/k/v projections
    # - RMSNorm on q and k before attention
    # - RoPE on q and k (optionally partial)
    # - causal masked SDPA
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rope_dims = rope_dims if rope_dims > 0 else self.head_dim
        if self.rope_dims % 2 != 0:
            raise ValueError("rope_dims must be even")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.rope_dims, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = rms_norm(q).astype(COMPUTE_DTYPE)
        k = rms_norm(k).astype(COMPUTE_DTYPE)
        if self.rope_dims < self.head_dim:
            q_rope, q_pass = q[..., :self.rope_dims], q[..., self.rope_dims:]
            k_rope, k_pass = k[..., :self.rope_dims], k[..., self.rope_dims:]
            q = mx.concatenate([self.rope(q_rope), q_pass], axis=-1)
            k = mx.concatenate([self.rope(k_rope), k_pass], axis=-1)
        else:
            q = self.rope(q)
            k = self.rope(k)
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # Baseline MLP uses relu^2 instead of GELU/SiLU. It is cheap and works well in this setup.
    def __init__(self, dim: int, mlp_mult: int, mlp_hidden: int = 0):
        super().__init__()
        hidden = mlp_hidden if mlp_hidden > 0 else dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


def hashed_context_ids(input_ids: mx.array, buckets: int, order: int) -> mx.array:
    if buckets <= 0 or order < 2:
        return mx.zeros(input_ids.shape, dtype=mx.int32)
    ids = input_ids.astype(mx.int64) + 1
    bsz, seqlen = input_ids.shape
    if order == 2 and seqlen > 1:
        hashed = ((ids[:, :-1] * 1315423911) ^ (ids[:, 1:] * 2654435761)) % buckets
        pad = mx.zeros((bsz, 1), dtype=mx.int64)
        return mx.concatenate((pad, hashed), axis=1).astype(mx.int32)
    if order == 3 and seqlen > 2:
        hashed = ((ids[:, :-2] * 73856093) ^ (ids[:, 1:-1] * 19349663) ^ (ids[:, 2:] * 83492791)) % buckets
        pad = mx.zeros((bsz, 2), dtype=mx.int64)
        return mx.concatenate((pad, hashed), axis=1).astype(mx.int32)
    return mx.zeros(input_ids.shape, dtype=mx.int32)


class HashContinuationExpert(nn.Module):
    def __init__(
        self,
        dim: int,
        hash_dim: int,
        bigram_buckets: int,
        trigram_buckets: int,
        use_gate: bool,
        mixer_hidden: int,
    ):
        super().__init__()
        self.bigram_buckets = bigram_buckets
        self.trigram_buckets = trigram_buckets
        self.use_gate = use_gate
        self.mixer_hidden = mixer_hidden
        self.bigram_emb = nn.Embedding(bigram_buckets, hash_dim) if bigram_buckets > 0 else None
        self.trigram_emb = nn.Embedding(trigram_buckets, hash_dim) if trigram_buckets > 0 else None
        feat_dim = hash_dim * int(self.bigram_emb is not None) + hash_dim * int(self.trigram_emb is not None)
        self.proj = CastedLinear(feat_dim, dim) if feat_dim > 0 else None
        gate_in_dim = dim * 2
        self.gate_fc = CastedLinear(gate_in_dim, mixer_hidden) if use_gate and mixer_hidden > 0 and self.proj is not None else None
        self.gate_proj = CastedLinear(mixer_hidden if mixer_hidden > 0 else gate_in_dim, dim) if use_gate and self.proj is not None else None
        if self.proj is not None:
            self.proj.weight = (mx.random.normal(self.proj.weight.shape, dtype=mx.float32) * 0.02).astype(COMPUTE_DTYPE)
        if self.gate_proj is not None:
            self.gate_proj.weight = mx.zeros_like(self.gate_proj.weight)

    def __call__(self, input_ids: mx.array, base_x: mx.array) -> mx.array:
        feats: list[mx.array] = []
        if self.bigram_emb is not None:
            feats.append(self.bigram_emb(hashed_context_ids(input_ids, self.bigram_buckets, 2)).astype(COMPUTE_DTYPE))
        if self.trigram_emb is not None:
            feats.append(self.trigram_emb(hashed_context_ids(input_ids, self.trigram_buckets, 3)).astype(COMPUTE_DTYPE))
        if not feats or self.proj is None:
            return mx.zeros_like(base_x)
        x = feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=-1)
        x = self.proj(x).astype(COMPUTE_DTYPE)
        if self.gate_proj is not None:
            gate_in = mx.concatenate((base_x.astype(COMPUTE_DTYPE), x), axis=-1)
            if self.gate_fc is not None:
                gate_in = nn.relu(self.gate_fc(gate_in)).astype(COMPUTE_DTYPE)
            x = x * mx.sigmoid(self.gate_proj(gate_in).astype(x.dtype))
        return x


class BigramHashEmbedding(nn.Module):
    def __init__(self, num_buckets: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, bigram_dim)
        self.embed.weight = mx.zeros_like(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim) if bigram_dim != model_dim else None
        if self.proj is not None:
            self.proj.weight = mx.zeros_like(self.proj.weight)
        self.scale = mx.array(0.05, dtype=mx.float32)

    def hash_ids(self, input_ids: mx.array) -> mx.array:
        ids = input_ids.astype(mx.int32)
        mod = self.num_buckets - 1
        out = mx.zeros_like(ids)
        out[:, 0] = mod
        out[:, 1:] = mx.bitwise_xor(36313 * ids[:, 1:], 27191 * ids[:, :-1]) % mod
        return out.astype(mx.int32)

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.embed(self.hash_ids(input_ids)).astype(COMPUTE_DTYPE)
        if self.proj is not None:
            h = self.proj(h).astype(COMPUTE_DTYPE)
        return h * self.scale.astype(h.dtype)


class DynamicHashMemoryExpert(nn.Module):
    def __init__(
        self,
        dim: int,
        memory_dim: int,
        buckets: int,
        recent_weight: float,
        use_gate: bool,
        chunk_size: int,
        min_count: int,
        count_alpha: float,
    ):
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim
        self.buckets = buckets
        self.recent_weight = float(np.clip(recent_weight, 0.0, 1.0))
        self.use_gate = use_gate
        self.chunk_size = max(int(chunk_size), 1)
        self.min_count = max(int(min_count), 0)
        self.count_alpha = max(float(count_alpha), 0.0)
        self.value_proj = CastedLinear(dim, memory_dim) if buckets > 0 else None
        self.out_proj = CastedLinear(memory_dim * 2, dim) if buckets > 0 else None
        self.gate_proj = CastedLinear(dim * 2, 1) if use_gate and buckets > 0 else None
        if self.out_proj is not None:
            self.out_proj.weight = (mx.random.normal(self.out_proj.weight.shape, dtype=mx.float32) * 0.02).astype(COMPUTE_DTYPE)
        if self.gate_proj is not None:
            self.gate_proj.weight = mx.zeros_like(self.gate_proj.weight)

    def __call__(self, input_ids: mx.array, target_ids: mx.array | None, base_x: mx.array, tok_emb_weight: mx.array) -> mx.array:
        if self.buckets <= 0 or self.value_proj is None or self.out_proj is None or target_ids is None:
            return mx.zeros_like(base_x)
        bsz, seqlen = input_ids.shape
        ctx_ids = hashed_context_ids(input_ids, self.buckets, 2)
        tgt_emb = rms_norm(tok_emb_weight[target_ids].astype(COMPUTE_DTYPE))
        values = self.value_proj(tgt_emb).astype(COMPUTE_DTYPE)
        mem_sum = mx.zeros((bsz, self.buckets, self.memory_dim), dtype=COMPUTE_DTYPE)
        mem_count = mx.zeros((bsz, self.buckets, 1), dtype=mx.float32)
        mem_recent = mx.zeros((bsz, self.buckets, self.memory_dim), dtype=COMPUTE_DTYPE)
        batch_idx = mx.arange(bsz, dtype=mx.int32)
        outs: list[mx.array] = []
        one = mx.ones((bsz, 1), dtype=mx.float32)
        batch_grid = mx.arange(bsz, dtype=mx.int32)[:, None]
        for s in range(0, seqlen, self.chunk_size):
            e = min(s + self.chunk_size, seqlen)
            bucket = ctx_ids[:, s:e]
            recent = mem_recent[batch_grid, bucket]
            count = mem_count[batch_grid, bucket]
            avg = mem_sum[batch_grid, bucket] / mx.maximum(count, 1.0)
            if self.recent_weight > 0.0:
                mixed = self.recent_weight * recent + (1.0 - self.recent_weight) * avg
                mem_feat = mx.concatenate((mixed, avg), axis=-1)
            else:
                mem_feat = mx.concatenate((recent, avg), axis=-1)
            mem_out = self.out_proj(mem_feat).astype(COMPUTE_DTYPE)
            if self.gate_proj is not None:
                gate_in = mx.concatenate((base_x[:, s:e].astype(COMPUTE_DTYPE), mem_out), axis=-1)
                mem_out = mem_out * mx.sigmoid(self.gate_proj(gate_in).astype(COMPUTE_DTYPE))
            if self.min_count > 0:
                active = (count >= float(self.min_count)).astype(mem_out.dtype)
                mem_out = mem_out * active
            if self.count_alpha > 0.0:
                conf = count.astype(mem_out.dtype) / (count.astype(mem_out.dtype) + self.count_alpha)
                mem_out = mem_out * conf
            outs.append(mem_out)
            val_chunk = values[:, s:e]
            batch_flat = mx.repeat(batch_idx, e - s)
            bucket_flat = bucket.reshape(-1)
            val_flat = val_chunk.reshape(-1, self.memory_dim)
            mem_sum = mem_sum.at[batch_flat, bucket_flat].add(val_flat)
            mem_count = mem_count.at[batch_flat, bucket_flat].add(mx.ones((batch_flat.shape[0], 1), dtype=mx.float32))
            for t in range(s, e):
                bucket_t = ctx_ids[:, t]
                val_t = values[:, t]
                old_recent = mem_recent[batch_idx, bucket_t]
                mem_recent = mem_recent.at[batch_idx, bucket_t].add(val_t - old_recent)
        return mx.concatenate(outs, axis=1) if outs else mx.zeros_like(base_x)


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        g = mx.sigmoid(self.gate.astype(x.dtype))[None, None, :]
        x_prev = mx.concatenate((mx.zeros_like(x[:, :1]), x[:, :-1]), axis=1)
        return (1.0 - g) * x + g * x_prev


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        mlp_hidden: int = 0,
        rope_dims: int = 0,
        ln_scale_factor: float = 1.0,
    ):
        super().__init__()
        self.ln_scale_factor = ln_scale_factor
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=rope_dims)
        self.mlp = MLP(dim, mlp_mult, mlp_hidden=mlp_hidden)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        s = self.ln_scale_factor
        attn_out = self.attn(self.attn_norm(x) * s)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * s)
        return x


class GPT(nn.Module):
    # - token embedding + RMSNorm
    # - encoder half accumulates skip tensors
    # - decoder half consumes reversed skips with learned skip_weights
    # - tied embeddings for the LM head (the baseline default setup)
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 logit_chunk_tokens: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float, mlp_hidden: int = 0, rope_dims: int = 0, ln_scale: bool = False,
                 bigram_hash_buckets: int = 0, trigram_hash_buckets: int = 0,
                 hash_dim: int = 64, hash_gate: bool = True, hash_mixer_hidden: int = 0,
                 hash_memory_buckets: int = 0, hash_memory_dim: int = 64,
                 hash_memory_recent_weight: float = 0.2, hash_memory_gate: bool = True,
                 hash_memory_chunk_size: int = 1, hash_memory_min_count: int = 0,
                 hash_memory_mode: str = "input", hash_memory_count_alpha: float = 0.0,
                 hash_memory_scale: float = 1.0, smear_gate: bool = False, simple_bigram: bool = False):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.hash_memory_mode = hash_memory_mode
        self.hash_memory_scale = hash_memory_scale

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.bigram = BigramHashEmbedding(bigram_hash_buckets, hash_dim, dim) if simple_bigram and bigram_hash_buckets > 0 else None
        self.hash_expert = None if self.bigram is not None else HashContinuationExpert(
            dim, hash_dim, bigram_hash_buckets, trigram_hash_buckets, hash_gate, hash_mixer_hidden
        )
        self.hash_memory_expert = DynamicHashMemoryExpert(
            dim, hash_memory_dim, hash_memory_buckets, hash_memory_recent_weight, hash_memory_gate,
            hash_memory_chunk_size, hash_memory_min_count, hash_memory_count_alpha
        )
        self.smear = SmearGate(dim) if smear_gate else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, mlp_hidden=mlp_hidden,
                  rope_dims=rope_dims,
                  ln_scale_factor=(1.0 / math.sqrt(i + 1)) if ln_scale else 1.0)
            for i in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array, target_ids: mx.array | None = None) -> mx.array:
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        if self.bigram is not None:
            x = x + self.bigram(input_ids).astype(x.dtype)
        base_x = rms_norm(x)
        if self.hash_expert is not None and self.hash_expert.proj is not None:
            x = x + self.hash_expert(input_ids, base_x).astype(x.dtype)
        x = rms_norm(x)
        if self.smear is not None:
            x = self.smear(x).astype(x.dtype)
        mem_bias = None
        if self.hash_memory_expert.out_proj is not None:
            mem_bias = (self.hash_memory_scale * self.hash_memory_expert(input_ids, target_ids, x, self.tok_emb.weight)).astype(x.dtype)
            if self.hash_memory_mode == "input":
                x = x + mem_bias
        x0 = x
        skips: list[mx.array] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            # Odd layer counts have one more decoder block than encoder block. The baseline only
            # applies a skip connection when one exists, then runs the remaining decoder block(s)
            # without an added skip.
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        if mem_bias is not None and self.hash_memory_mode == "output":
            x = x + mem_bias
        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        # Cross-entropy over flattened tokens. We keep optional logit chunking because it is a useful
        # memory knob on Macs, but the common path is chunk_tokens=0 (single matmul + CE).
        x = self(input_ids, target_ids=target_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)

# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================
class Muon:
    # Muon applies SGD-momentum to matrix gradients, then orthogonalizes the result before the
    # parameter update.
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    # - embeddings: Adam with the tied-embedding LR
    # - block matrices (2D): Muon
    # - block scalars + skip weights: Adam
    # This preserves the high-level optimization behavior even though MLX internals differ.
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_keys = [k for k in ("tok_emb.weight", "bigram.embed.weight") if k in params]
        self.matrix_keys = [
            k
            for k, p in params.items()
            if k not in self.embed_keys and p.ndim == 2 and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k
            for k, p in params.items()
            if k not in self.embed_keys and (p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS))
        ]

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        updated.update(
            self.adam_embed.apply_gradients(
                {k: grads[k] for k in self.embed_keys},
                {k: params[k] for k in self.embed_keys},
            )
        )

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (INT8 + ZLIB)
# ==============================================================================
# - per-row int8 for 2D float tensors
# - per-tensor int8 for other float tensors
# - fp16 passthrough for small float tensors
# - exact passthrough for non-floats

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int8(flat_state: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP16_NAME_PATTERNS):
            passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
            kept = np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            # Broadcast the saved row scale back across trailing dimensions.
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def shard_indices(dataset_dir: Path, split: str) -> list[int]:
    indices: list[int] = []
    for path in sorted(dataset_dir.glob(f"fineweb_{split}_*.bin")):
        suffix = path.stem.rsplit("_", 1)[-1]
        if not suffix.isdigit():
            raise ValueError(f"Unexpected shard filename: {path.name}")
        indices.append(int(suffix))
    return indices


def validate_shard_layout(
    dataset_dir: Path,
    *,
    split: str,
    expected_files: int | None,
    allow_prefix_subset: bool,
) -> int:
    indices = shard_indices(dataset_dir, split)
    if not indices:
        raise FileNotFoundError(f"No {split} shards found in {dataset_dir}")
    if indices != list(range(len(indices))):
        raise ValueError(f"{dataset_dir.name} has non-contiguous {split} shard indices: {indices[:8]}...")
    if expected_files is None:
        return len(indices)
    if allow_prefix_subset:
        if len(indices) > expected_files:
            raise ValueError(f"{dataset_dir.name} has too many {split} shards: found {len(indices)}, expected {expected_files}")
        return len(indices)
    if len(indices) != expected_files:
        raise ValueError(f"{dataset_dir.name} must use the full {split} split: found {len(indices)}, expected {expected_files}")
    return len(indices)


def validate_dataset_tokenizer_pair(
    *,
    data_path: str,
    tokenizer_path: str,
    sp: spm.SentencePieceProcessor,
    vocab_size: int,
    require_manifest: bool,
    require_full_val_split: bool,
) -> dict[str, object]:
    dataset_dir = Path(data_path).resolve()
    tokenizer_file = Path(tokenizer_path).resolve()
    info: dict[str, object] = {
        "dataset_name": dataset_dir.name,
        "manifest_path": None,
        "tokenizer_name": None,
        "tokenizer_kind": None,
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "docs_sha256": None,
        "actual_train_files": len(list(dataset_dir.glob("fineweb_train_*.bin"))),
        "expected_train_files": None,
        "actual_val_files": len(list(dataset_dir.glob("fineweb_val_*.bin"))),
        "expected_val_files": None,
        "expected_docs_val": None,
        "bos_id": int(sp.bos_id()),
        "eos_id": int(sp.eos_id()),
    }
    if int(sp.vocab_size()) != vocab_size:
        raise ValueError(f"VOCAB_SIZE={vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")

    manifest_path = dataset_dir.parents[1] / "manifest.json" if len(dataset_dir.parents) >= 2 else None
    if manifest_path is None or not manifest_path.is_file():
        if require_manifest:
            raise FileNotFoundError(f"manifest.json is required for {dataset_dir}, got none at {manifest_path}")
        return info

    info["manifest_path"] = str(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    info["docs_sha256"] = ((manifest.get("docs_meta") or {}).get("docs_sha256"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        raise ValueError(f"{dataset_dir.name} not found in {manifest_path}")
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    if tokenizer_name and tokenizer_entry is None:
        raise ValueError(f"Tokenizer {tokenizer_name} for {dataset_dir.name} is missing from {manifest_path}")
    info["tokenizer_name"] = tokenizer_name
    info["tokenizer_kind"] = dataset_entry.get("tokenizer_kind")
    if dataset_entry.get("vocab_size") is not None and int(dataset_entry["vocab_size"]) != vocab_size:
        raise ValueError(f"{dataset_dir.name} expects vocab_size={dataset_entry['vocab_size']}, got {vocab_size}")
    if dataset_entry.get("bos_id") is not None and int(dataset_entry["bos_id"]) != int(sp.bos_id()):
        raise ValueError(f"{dataset_dir.name} expects bos_id={dataset_entry['bos_id']}, got {sp.bos_id()}")
    if dataset_entry.get("eos_id") is not None and int(dataset_entry["eos_id"]) != int(sp.eos_id()):
        raise ValueError(f"{dataset_dir.name} expects eos_id={dataset_entry['eos_id']}, got {sp.eos_id()}")
    info["bos_id"] = int(dataset_entry.get("bos_id", sp.bos_id()))
    info["eos_id"] = int(dataset_entry.get("eos_id", sp.eos_id()))

    expected_model_rel = (tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path")
    expected_name = Path(expected_model_rel or "").name
    if expected_name and tokenizer_file.name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {tokenizer_file.name}")
    if expected_model_rel:
        expected_model_path = (manifest_path.parent / expected_model_rel).resolve()
        if expected_model_path.is_file():
            expected_sha256 = sha256_file(expected_model_path)
            if expected_sha256 != info["tokenizer_sha256"]:
                raise ValueError(
                    f"{dataset_dir.name} tokenizer content hash mismatch: expected {expected_sha256}, got {info['tokenizer_sha256']}"
                )
            expected_sp = spm.SentencePieceProcessor(model_file=str(expected_model_path))
            for field in ("bos_id", "eos_id", "pad_id", "unk_id"):
                if getattr(expected_sp, field)() != getattr(sp, field)():
                    raise ValueError(f"{dataset_dir.name} tokenizer {field} mismatch against manifest model")
            for token_id in range(int(sp.vocab_size())):
                if expected_sp.id_to_piece(token_id) != sp.id_to_piece(token_id):
                    raise ValueError(f"{dataset_dir.name} tokenizer piece mismatch at id={token_id}")

    stats = dataset_entry.get("stats") or {}
    if stats.get("files_train") is not None:
        info["expected_train_files"] = int(stats["files_train"])
    if stats.get("files_val") is not None:
        info["expected_val_files"] = int(stats["files_val"])
    if stats.get("docs_val") is not None:
        info["expected_docs_val"] = int(stats["docs_val"])
    info["actual_train_files"] = validate_shard_layout(
        dataset_dir,
        split="train",
        expected_files=info["expected_train_files"],
        allow_prefix_subset=True,
    )
    info["actual_val_files"] = validate_shard_layout(
        dataset_dir,
        split="val",
        expected_files=info["expected_val_files"],
        allow_prefix_subset=not require_full_val_split,
    )
    return info


def validate_bos_consistency(val_tokens: np.ndarray, *, bos_id: int, expected_docs_val: int | None) -> None:
    if val_tokens.size == 0:
        raise ValueError("Validation tokens are empty")
    if int(val_tokens[0]) != bos_id:
        raise ValueError(f"Validation split must start with bos_id={bos_id}, got {int(val_tokens[0])}")
    bos_positions = np.flatnonzero(val_tokens == bos_id)
    if expected_docs_val is not None and int(bos_positions.size) != expected_docs_val:
        raise ValueError(f"Expected {expected_docs_val} BOS markers in validation, got {int(bos_positions.size)}")
    if bos_positions.size > 1 and np.any(np.diff(bos_positions) < 2):
        raise ValueError("Validation document boundaries are malformed: adjacent BOS markers are too close")


def git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")


def load_validation_tokens_raw(pattern: str) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    return np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))


def truncate_validation_tokens(tokens: np.ndarray, seq_len: int) -> np.ndarray:
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    return truncate_validation_tokens(load_validation_tokens_raw(pattern), seq_len)


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
) -> tuple[mx.array, dict]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)  # materialize each chunk to cap peak memory
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(
    args: Hyperparameters,
    compiled_loss,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = compiled_loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb

# -----------------------------
# TRAINING
# -----------------------------

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(_np_float32(grad)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def main() -> None:
    # ==============================================================================
    # TOKENIZER + VALIDATION METRIC SETUP
    # ==============================================================================
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    metrics_path = out_dir / "runs.jsonl"
    git_sha = git_commit_sha()
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tie_embeddings:
        raise NotImplementedError("train_gpt_mlx.py only supports tied embeddings")
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    dataset_info = validate_dataset_tokenizer_pair(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        sp=sp,
        vocab_size=args.vocab_size,
        require_manifest=args.require_data_manifest,
        require_full_val_split=args.require_full_val_split,
    )
    raw_val_tokens = load_validation_tokens_raw(args.val_files)
    validate_bos_consistency(
        raw_val_tokens,
        bos_id=int(dataset_info["bos_id"]),
        expected_docs_val=dataset_info["expected_docs_val"],
    )
    val_tokens = truncate_validation_tokens(raw_val_tokens, args.train_seq_len)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    # ==============================================================================
    # TRAINING SETUP
    # ==============================================================================
    mx.random.seed(args.seed)

    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=str(dataset_info["dataset_name"]))

    # ==============================================================================
    # MODEL + OPTIMIZER SETUP
    # ==============================================================================
    model = GPT(
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
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
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
        smear_gate=args.smear_gate,
        simple_bigram=args.simple_bigram,
    )
    opt = SplitOptimizers(model, args)

    # ==============================================================================
    # COMPILED TRAIN / EVAL FUNCTIONS (MLX)
    # ==============================================================================
    # The crucial MLX detail is capture scope: this model contains non-trainable arrays too (for example
    # inside RoPE modules), so compiling only against trainable parameters throws "uncaptured inputs".
    # Compiling the model-bound functions and capturing the full model state fixes that while still
    # returning gradients only for trainable parameters via nn.value_and_grad(...).
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )

    # Print config once so logs are self-describing.
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    log(
        f"dataset_manifest:path={dataset_info['manifest_path']} docs_sha256:{dataset_info['docs_sha256']} "
        f"tokenizer_name:{dataset_info['tokenizer_name']} tokenizer_kind:{dataset_info['tokenizer_kind']}"
    )
    log(
        f"tokenizer_identity:sha256={dataset_info['tokenizer_sha256']} "
        f"bos_id:{dataset_info['bos_id']} eos_id:{dataset_info['eos_id']}"
    )
    if dataset_info["expected_train_files"] is None:
        log(f"train_loader:dataset:{dataset_info['dataset_name']} train_shards:{dataset_info['actual_train_files']}")
    elif dataset_info["actual_train_files"] < dataset_info["expected_train_files"]:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_info['dataset_name']} "
            f"train_shards:{dataset_info['actual_train_files']}/{dataset_info['expected_train_files']} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(
            f"train_loader:dataset:{dataset_info['dataset_name']} "
            f"train_shards:{dataset_info['actual_train_files']}/{dataset_info['expected_train_files']}"
        )
    log(
        f"val_loader:dataset:{dataset_info['dataset_name']} "
        f"val_shards:{dataset_info['actual_val_files']}/{dataset_info['expected_val_files']}"
    )
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings} "
        f"bigram_hash_buckets:{args.bigram_hash_buckets} trigram_hash_buckets:{args.trigram_hash_buckets} "
        f"hash_dim:{args.hash_dim} hash_gate:{int(args.hash_gate)} hash_mixer_hidden:{args.hash_mixer_hidden} "
        f"hash_memory_buckets:{args.hash_memory_buckets} hash_memory_dim:{args.hash_memory_dim} "
        f"hash_memory_recent_weight:{args.hash_memory_recent_weight:.3f} hash_memory_gate:{int(args.hash_memory_gate)} "
        f"hash_memory_chunk_size:{args.hash_memory_chunk_size} hash_memory_min_count:{args.hash_memory_min_count} "
        f"hash_memory_mode:{args.hash_memory_mode} hash_memory_count_alpha:{args.hash_memory_count_alpha:.3f} "
        f"hash_memory_scale:{args.hash_memory_scale:.3f} smear_gate:{int(args.smear_gate)} "
        f"simple_bigram:{int(args.simple_bigram)}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(f"final_eval_mode:flat bos_id:{dataset_info['bos_id']}")
    log(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"linear_weight:{model.blocks[0].attn.c_q.weight.dtype} "
        f"skip_weights:{model.skip_weights.dtype}"
    )

    # ==============================================================================
    # TRAINING LOOP
    # ==============================================================================
    if args.warmup_steps > 0:
        # Warmup should only prime MLX compile/allocation paths. Updating parameters here forces us
        # to snapshot and restore model/optimizer state, which is expensive on unified-memory Macs.
        # Instead we run the real train shapes, force the loss/grads to materialize, and then reset
        # the loader so measured training still starts from the true init and token window.
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime the standalone eval graph once too. It is compiled separately from value_and_grad.
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        if val_batch_tokens < args.train_seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                f"TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
        x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
        warm_val_loss = compiled_loss(x_val, y_val)
        mx.eval(warm_val_loss)
        mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=str(dataset_info["dataset_name"]))

    train_time_ms = 0.0
    last_plain_val_loss: float | None = None
    last_plain_val_bpb: float | None = None
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Validation always scans the same fixed full validation split.
            val_loss, val_bpb = eval_val(
                args,
                compiled_loss,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log,
            )
            last_plain_val_loss = val_loss
            last_plain_val_bpb = val_bpb
            if step % 25 == 0 or last_step:
                log(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)  # materialize each microbatch to cap peak memory

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # ==============================================================================
    # FINAL SERIALIZATION + QUANTIZED ROUNDTRIP EVAL
    # ==============================================================================
    # We always write a raw artifact and a quantized artifact, then validate the
    # quantized roundtrip directly by loading the dequantized tensors back into the
    # model and running one final validation pass.
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log(
        f"serialized_model_int8_zlib:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
    )

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_loss,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    append_jsonl(
        metrics_path,
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_sha": git_sha,
            "run_id": args.run_id,
            "script": "train_gpt_mlx.py",
            "data_path": args.data_path,
            "tokenizer_name": dataset_info["tokenizer_name"],
            "tokenizer_sha256": dataset_info["tokenizer_sha256"],
            "train_shards": dataset_info["actual_train_files"],
            "val_shards": dataset_info["actual_val_files"],
            "model_params": n_params,
            "train_seq_len": args.train_seq_len,
            "eval_seq_len": args.train_seq_len,
            "eval_stride": 0,
            "plain_val_loss": last_plain_val_loss,
            "plain_val_bpb": last_plain_val_bpb,
            "roundtrip_val_loss": q_val_loss,
            "roundtrip_val_bpb": q_val_bpb,
            "final_mode": "flat",
            "final_val_loss": q_val_loss,
            "final_val_bpb": q_val_bpb,
            "train_time_ms": train_time_ms,
            "eval_time_ms": q_eval_ms,
            "raw_model_bytes": out_path.stat().st_size,
            "quant_bytes": quant_file_bytes,
            "code_bytes": len(code.encode("utf-8")),
            "total_bytes": quant_file_bytes + len(code.encode("utf-8")),
        },
    )


if __name__ == "__main__":
    main()
