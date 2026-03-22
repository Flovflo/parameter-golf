#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import download_hf_docs_and_tokenize as tokmod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train tokenizer candidates on sampled docs and compare token counts.")
    p.add_argument("--docs-jsonl", required=True)
    p.add_argument("--tokenizer-config", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--tokenizer-train-docs", type=int, default=12000)
    p.add_argument("--eval-docs", type=int, default=4000)
    p.add_argument("--skip-byte", action="store_true")
    return p.parse_args()


def load_eval_docs(path: Path, eval_docs: int) -> list[str]:
    docs = list(tokmod.iter_docs(path))
    if len(docs) < eval_docs + 100:
        raise ValueError(f"need more sampled docs for a stable eval split, got {len(docs)}")
    return docs[-eval_docs:]


def encode_len(tok: dict, texts: list[str]) -> int:
    batch_encode = tok.get("encode_batch")
    if callable(batch_encode):
        batches = batch_encode(texts)
    else:
        batches = [tok["encode"](text) for text in texts]
    return sum(len(item) + 1 for item in batches)


def tokenizer_artifact_bytes(manifest: dict) -> int:
    total = 0
    for key in ("model_path", "vocab_path", "path"):
        value = manifest.get(key)
        if value:
            total += Path(value).stat().st_size
    return total


def print_table(rows: list[dict]) -> None:
    cols = ["name", "vocab", "bytes_per_tok", "tokens", "tok_per_byte", "delta_pct", "artifact_kb"]
    widths = {c: len(c) for c in cols}
    for row in rows:
        widths["name"] = max(widths["name"], len(str(row["name"])))
        widths["vocab"] = max(widths["vocab"], len(str(row["vocab"])))
        widths["bytes_per_tok"] = max(widths["bytes_per_tok"], len(f"{row['bytes_per_tok']:.4f}"))
        widths["tokens"] = max(widths["tokens"], len(str(row["tokens"])))
        widths["tok_per_byte"] = max(widths["tok_per_byte"], len(f"{row['tok_per_byte']:.6f}"))
        widths["delta_pct"] = max(widths["delta_pct"], len(f"{row['delta_pct']:+.2f}"))
        widths["artifact_kb"] = max(widths["artifact_kb"], len(f"{row['artifact_kb']:.1f}"))
    header = " ".join(f"{c:>{widths[c]}}" for c in cols)
    print(header)
    print(" ".join("-" * widths[c] for c in cols))
    for row in rows:
        print(
            " ".join(
                [
                    f"{row['name']:>{widths['name']}}",
                    f"{row['vocab']:>{widths['vocab']}}",
                    f"{row['bytes_per_tok']:.4f}".rjust(widths["bytes_per_tok"]),
                    f"{row['tokens']:>{widths['tokens']}}",
                    f"{row['tok_per_byte']:.6f}".rjust(widths["tok_per_byte"]),
                    f"{row['delta_pct']:+.2f}".rjust(widths["delta_pct"]),
                    f"{row['artifact_kb']:.1f}".rjust(widths["artifact_kb"]),
                ]
            )
        )


def main() -> None:
    args = parse_args()
    docs_path = Path(args.docs_jsonl).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    tokenizers_dir = output_root / "tokenizers"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)

    specs = tokmod.load_specs(Path(args.tokenizer_config).expanduser().resolve())
    tokenizers, selected_specs = tokmod.build_tokenizers(
        specs=specs,
        docs_jsonl=docs_path,
        tokenizers_dir=tokenizers_dir,
        tokenizer_train_docs=args.tokenizer_train_docs,
        skip_byte=args.skip_byte,
        reuse_sp_models={},
    )
    eval_docs = load_eval_docs(docs_path, args.eval_docs)
    eval_bytes = sum(len(doc.encode("utf-8")) for doc in eval_docs)

    rows: list[dict] = []
    baseline_tokens = None
    for tok, spec in zip(tokenizers, selected_specs, strict=True):
        total_tokens = encode_len(tok, eval_docs)
        if baseline_tokens is None:
            baseline_tokens = total_tokens
        rows.append(
            {
                "name": tok["name"],
                "vocab": tok["vocab_size"],
                "bytes_per_tok": eval_bytes / total_tokens,
                "tokens": total_tokens,
                "tok_per_byte": total_tokens / eval_bytes,
                "delta_pct": 100.0 * (total_tokens / baseline_tokens - 1.0),
                "artifact_kb": tokenizer_artifact_bytes(tok["manifest"]) / 1024.0,
                "source_spec": spec,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "tokenizer_proxy_results.json"
    results_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print_table(rows)
    print(f"results={results_path}")


if __name__ == "__main__":
    main()
