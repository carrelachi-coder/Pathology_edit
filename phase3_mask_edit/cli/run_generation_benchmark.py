#!/usr/bin/env python3
"""Launch configured paired-generation baselines on a remote server."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

from phase3_mask_edit.benchmark.generation_models import (
    load_generation_model_configs,
)


DEFAULT_CONFIG_DIR = Path("benchmark_configs/models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--host", default="amax2")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--log-root", default="/data1/zhao/wqx/benchmarks/logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = load_generation_model_configs(args.config_dir)
    selected = args.models or list(configs)
    missing = sorted(set(selected) - set(configs))
    if missing:
        raise ValueError(f"Unknown model IDs: {missing}")

    for model_id in selected:
        config = configs[model_id]
        if config.is_reused:
            print(
                f"[{model_id}] reuse: {config.execution.get('output_root')}",
                flush=True,
            )
            continue
        remote_command = config.build_remote_command(
            manifest=args.manifest,
            output_root=args.output_root,
            device=args.device,
            max_items=args.max_items,
            overwrite=args.overwrite,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            metadata_only=args.metadata_only,
        )
        if args.cuda_visible_devices:
            remote_command = (
                "export CUDA_VISIBLE_DEVICES="
                f"{shlex.quote(args.cuda_visible_devices)} && {remote_command}"
            )
        if args.detach:
            shard_suffix = (
                f"_shard{args.shard_index}of{args.num_shards}"
                if args.num_shards != 1
                else ""
            )
            log_path = (
                f"{args.log_root.rstrip('/')}/{model_id}{shard_suffix}.log"
            )
            remote_command = (
                f"mkdir -p {shlex.quote(args.log_root)} && "
                f"setsid -f sh -lc {shlex.quote(remote_command)} "
                f"> {shlex.quote(log_path)} 2>&1 < /dev/null"
            )
        print(f"[{model_id}] ssh {args.host} {shlex.quote(remote_command)}", flush=True)
        if args.execute:
            subprocess.run(["ssh", args.host, remote_command], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
