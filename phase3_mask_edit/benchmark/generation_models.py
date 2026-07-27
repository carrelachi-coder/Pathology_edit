"""Configuration contracts for the paired generation benchmark models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex

import yaml


@dataclass(frozen=True)
class GenerationModelConfig:
    path: Path
    model_id: str
    display_name: str
    model_type: str
    role: str
    allowed_inputs: tuple[str, ...]
    forbidden_inputs: tuple[str, ...]
    execution: dict

    @property
    def is_reused(self) -> bool:
        return self.execution.get("mode") == "reuse"

    def build_remote_command(
        self,
        manifest: str,
        output_root: str,
        device: str,
        max_items: int | None = None,
        overwrite: bool = False,
        num_shards: int = 1,
        shard_index: int = 0,
        metadata_only: bool = False,
    ) -> str:
        if self.is_reused:
            return ""
        if self.execution.get("mode") != "ssh_batch":
            raise ValueError(f"Unsupported execution mode for {self.model_id}")
        command = ["/usr/bin/env"]
        command.extend(
            f"{key}={value}"
            for key, value in self.execution.get("environment", {}).items()
        )
        command.extend([
            str(self.execution["env_python"]),
            str(self.execution["worker"]),
            "--model-id",
            self.model_id,
            "--manifest",
            manifest,
            "--output-root",
            output_root,
            "--device",
            device,
        ])
        for key, value in self.execution.get("arguments", {}).items():
            flag = f"--{key}"
            if isinstance(value, bool):
                command.append(flag if value else f"--no-{key}")
            elif value is not None:
                command.extend([flag, str(value)])
        if max_items is not None:
            command.extend(["--max-items", str(max_items)])
        if num_shards != 1:
            command.extend(
                ["--num-shards", str(num_shards), "--shard-index", str(shard_index)]
            )
        if overwrite:
            command.append("--overwrite")
        if metadata_only:
            command.append("--metadata-only")
        cwd = self.execution.get("cwd")
        shell_command = shlex.join(command)
        if cwd:
            shell_command = f"cd {shlex.quote(str(cwd))} && {shell_command}"
        return shell_command


def load_generation_model_config(path: Path) -> GenerationModelConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract = payload["input_contract"]
    config = GenerationModelConfig(
        path=path,
        model_id=str(payload["model_id"]),
        display_name=str(payload["display_name"]),
        model_type=str(payload["model_type"]),
        role=str(payload["role"]),
        allowed_inputs=tuple(contract["allowed"]),
        forbidden_inputs=tuple(contract.get("forbidden", ())),
        execution=dict(payload["execution"]),
    )
    if "target_image" in config.allowed_inputs:
        raise ValueError(f"{path}: target_image cannot be a generation input")
    if "target_image" not in config.forbidden_inputs:
        raise ValueError(f"{path}: target_image must be explicitly forbidden")
    return config


def load_generation_model_configs(config_dir: Path) -> dict[str, GenerationModelConfig]:
    configs = {}
    for path in sorted(config_dir.glob("*.yaml")):
        if path.name.startswith("._"):
            continue
        config = load_generation_model_config(path)
        if config.model_id in configs:
            raise ValueError(f"Duplicate model_id: {config.model_id}")
        configs[config.model_id] = config
    return configs
