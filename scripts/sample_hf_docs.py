#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import hf_hub_download, hf_hub_url


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample docs_selected.jsonl via HTTP Range requests.")
    p.add_argument("--repo-id", default="willdepueoai/parameter-golf")
    p.add_argument("--remote-root", default="datasets")
    p.add_argument("--filename", default="docs_selected.jsonl")
    p.add_argument("--sidecar", default="docs_selected.source_manifest.json")
    p.add_argument("--output", required=True)
    p.add_argument("--spans", type=int, default=8)
    p.add_argument("--span-bytes", type=int, default=4 * 1024 * 1024)
    p.add_argument("--docs-per-span", type=int, default=2500)
    p.add_argument("--prefix-bytes", type=int, default=4096)
    return p.parse_args()


def load_total_bytes(repo_id: str, remote_root: str, sidecar_name: str) -> int:
    sidecar_path = hf_hub_download(
        repo_id=repo_id,
        filename=sidecar_name,
        subfolder=remote_root,
        repo_type="dataset",
    )
    payload = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    total_bytes = int(payload["docs_bytes"])
    if total_bytes <= 0:
        raise ValueError(f"invalid docs_bytes in sidecar: {total_bytes}")
    return total_bytes


def fetch_span(url: str, start: int, end: int) -> bytes:
    req = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def extract_docs(blob: bytes, *, trim_first: bool, docs_per_span: int) -> list[str]:
    text = blob.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    if trim_first and lines:
        lines = lines[1:]
    docs: list[str] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        doc = payload.get("text")
        if isinstance(doc, str) and doc:
            docs.append(doc)
        if len(docs) >= docs_per_span:
            break
    return docs


def span_offsets(total_bytes: int, spans: int, span_bytes: int) -> list[int]:
    if spans <= 1 or total_bytes <= span_bytes:
        return [0]
    last = max(total_bytes - span_bytes, 0)
    return [int(round(i * last / (spans - 1))) for i in range(spans)]


def main() -> None:
    args = parse_args()
    if args.spans <= 0:
        raise ValueError("--spans must be positive")
    if args.span_bytes <= 0 or args.docs_per_span <= 0 or args.prefix_bytes < 0:
        raise ValueError("span/doc sizes must be positive")

    total_bytes = load_total_bytes(args.repo_id, args.remote_root, args.sidecar)
    url = hf_hub_url(
        repo_id=args.repo_id,
        filename=args.filename,
        subfolder=args.remote_root,
        repo_type="dataset",
    )
    offsets = span_offsets(total_bytes, args.spans, args.span_bytes)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    docs_written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for offset in offsets:
            start = max(offset - args.prefix_bytes, 0)
            end = min(offset + args.span_bytes - 1, total_bytes - 1)
            docs = extract_docs(
                fetch_span(url, start, end),
                trim_first=start > 0,
                docs_per_span=args.docs_per_span,
            )
            for doc in docs:
                out.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
            docs_written += len(docs)
            print(
                f"span_offset={offset} fetched_bytes={end - start + 1} docs={len(docs)} total_docs={docs_written}",
                flush=True,
            )
    print(f"wrote {docs_written} docs to {output_path}", flush=True)


if __name__ == "__main__":
    main()
