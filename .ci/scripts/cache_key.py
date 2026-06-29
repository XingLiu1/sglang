#!/usr/bin/env python3
"""Compute a stable cache key for one auto-mr-perf bench run, or write a meta
file alongside the cached artifacts for debugging.

The key is a SHA-256 of a deterministic blob built from:
  role, commit, image, model_path, serve_args, serve_envs (sorted),
  bench_args, sha256(bench_script_file).

Pure stdlib so it runs on the bare build node without any package install.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


# Order matters for the hash. New fields must be appended at the end so an old
# cache entry rebuilt against a newer cache_key.py still produces a different
# key when the new field actually differs (which is what we want), but the same
# key when the new field is absent on both sides via the same default.
KEY_FIELDS = (
    "role",
    "commit",
    "image",
    "model_path",
    "serve_args",
    "serve_envs_sorted",
    "bench_args",
    "bench_script_sha256",
    "tacops_commit",
)


def _normalize_envs(envs: str) -> str:
    """Whitespace-split the KEY=VAL string, sort, rejoin with single spaces.

    The pipeline accepts envs as a free-form space-separated string. Two users
    typing the same set in different order should hit the same cache. We do
    NOT do this for serve_args / bench_args because positional flags can have
    semantic meaning (later flags override earlier ones in some CLIs).
    """
    return " ".join(sorted(envs.split()))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_blob(args) -> dict:
    return {
        "role": args.role,
        "commit": args.commit,
        "image": args.image,
        "model_path": args.model_path,
        "serve_args": args.serve_args,
        "serve_envs_sorted": _normalize_envs(args.serve_envs),
        "bench_args": args.bench_args,
        "bench_script_sha256": _sha256_file(Path(args.bench_script)),
        "tacops_commit": args.tacops_commit,
    }


def _hash_blob(blob: dict) -> str:
    # Stable serialization: KEY_FIELDS order, "key=value\n" lines.
    # Avoid json.dumps(sort_keys=True) so the field order is explicit and
    # immune to accidental dict reordering bugs.
    text = "".join(f"{k}={blob[k]}\n" for k in KEY_FIELDS)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cmd_compute(args) -> int:
    blob = _build_blob(args)
    digest = _hash_blob(blob)
    sys.stdout.write(digest[:16])
    return 0


def cmd_write_meta(args) -> int:
    blob = _build_blob(args)
    digest = _hash_blob(blob)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "key": digest[:16],
        "key_full": digest,
        "fields": blob,
        # Original (un-normalized) envs for human inspection.
        "serve_envs_raw": args.serve_envs,
        "bench_script_path": args.bench_script,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


def _add_field_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--role", required=True, choices=("prefill", "decode"))
    p.add_argument("--commit", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--serve-args", required=True)
    p.add_argument("--serve-envs", required=True)
    p.add_argument("--bench-args", required=True)
    p.add_argument("--bench-script", required=True)
    p.add_argument("--tacops-commit", required=True,
                   help="git rev-parse HEAD inside 3rdparty/tacops; varies per branch.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_compute = sub.add_parser("compute", help="Print 16-char hex cache key.")
    _add_field_args(p_compute)
    p_compute.set_defaults(fn=cmd_compute)

    p_write = sub.add_parser(
        "write-meta", help="Write meta.json into <out> for debugging."
    )
    _add_field_args(p_write)
    p_write.add_argument("--out", required=True)
    p_write.set_defaults(fn=cmd_write_meta)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
