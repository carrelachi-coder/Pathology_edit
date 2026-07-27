#!/usr/bin/env python3
"""Batch runner for Phase 3 UI mask-edit debug manifests.

The runner intentionally reuses the non-interactive backend functions from
``scripts.phase3_end_to_end_ui`` so batch runs follow the same organic_v2 contour
path as the Gradio UI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_DEFAULT_MANIFEST = (
    REPO_ROOT
    / "docs"
    / "superpowers"
    / "debug_sets"
    / "phase3_ui_mask_edit_debug_manifest.json"
)
REMOTE_DEFAULT_MANIFEST = Path(
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/dubug/phase3_ui_mask_edit_debug_manifest.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "phase3_manifest_pipeline"
POSIX_ROOT_ANCHORS = ("/data/", "/home/", "/mnt/", "/workspace/", "/scratch/")
PATCH_DIR_BY_DATASET = {
    "BCSS": "BCSS_PATCHES",
    "GlaS": "GlaS_PATCHES",
    "IGNITE": "IGNITE_PATCHES",
    "ORCA": "ORCA_PATCHES",
    "PANDA": "PANDA_PATCHES",
    "PUMA": "PUMA_PATCHES",
}
FIELD_SUBDIRS = {
    "source_image": ("images",),
    "source_tissue_mask": ("tissue_masks", "masks"),
    "source_nuclei_mask": ("nuclei_masks", "cell_masks", "cells"),
}


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    manifest = _load_json(args.manifest)
    runtime = _mapping(manifest.get("runtime"))
    runtime["defaults"] = _mapping(manifest.get("defaults"))
    dataset_roots = _dataset_roots(manifest, args)
    variants = _selected_variants(runtime, args.variants)
    cases = _selected_cases(manifest, args)

    batch_dir = args.output_root / (args.run_id or time.strftime("%Y%m%d_%H%M%S"))
    batch_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        {"case": case, "variant": variant}
        for case in cases
        for variant in variants
    ]
    batch_plan = {
        "manifest": str(args.manifest),
        "output_root": str(batch_dir),
        "case_count": len(cases),
        "variant_count": len(variants),
        "run_count": len(runs),
        "dataset_roots": {key: _path_text(value) for key, value in dataset_roots.items()},
        "runs": [
            {
                "case_id": item["case"].get("case_id"),
                "dataset": item["case"].get("dataset"),
                "variant_id": item["variant"].get("variant_id"),
                "edit_mode": item["variant"].get("edit_mode"),
            }
            for item in runs
        ],
    }
    _write_json(batch_plan, batch_dir / "batch_plan.json")
    if args.plan_only:
        print(json.dumps(batch_plan, indent=2, ensure_ascii=False))
        return 0

    ui = _load_ui_backend()
    summary: dict[str, Any] = {
        **batch_plan,
        "stop_after": args.stop_after,
        "results": [],
    }
    failed = 0
    for item in runs:
        case = item["case"]
        variant = item["variant"]
        case_id = str(case.get("case_id", "case"))
        variant_id = str(variant.get("variant_id") or variant.get("edit_mode") or "variant")
        run_dir = batch_dir / case_id / variant_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result_record: dict[str, Any] = {
            "case_id": case_id,
            "dataset": case.get("dataset"),
            "profile": case.get("profile"),
            "variant_id": variant_id,
            "edit_mode": variant.get("edit_mode"),
            "output_dir": _path_text(run_dir),
        }
        try:
            paths = _resolve_case_paths(case, dataset_roots, require_exists=True)
            _write_json({"case": case, "variant": variant, "paths": _stringify(paths)}, run_dir / "run_config.json")
            state = _prepare_state(ui, case, paths, run_dir)
            state, tissue_info = _run_tissue_stage(ui, state, case, variant, runtime, args)
            result_record["tissue"] = tissue_info

            if args.stop_after in {"cell", "generation"}:
                state, cell_info = _run_cell_stage(ui, state, case, runtime, args)
                result_record["cell"] = cell_info

            if args.stop_after == "generation":
                state, generation_info = _run_generation_stage(ui, state, case, runtime, args)
                result_record["generation"] = generation_info

            result_record["status"] = "completed"
        except Exception as exc:  # noqa: BLE001 - batch runner should record every failing case.
            failed += 1
            result_record["status"] = "failed"
            result_record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_json(result_record["error"], run_dir / "error.json")
            if args.fail_fast:
                summary["results"].append(result_record)
                _write_json(summary, batch_dir / "batch_summary.json")
                raise
        summary["results"].append(result_record)
        _write_json(summary, batch_dir / "batch_summary.json")
        print(
            f"[{result_record['status']}] {case_id} / {variant_id} -> {_path_text(run_dir)}",
            flush=True,
        )

    summary["completed"] = sum(1 for item in summary["results"] if item.get("status") == "completed")
    summary["failed"] = failed
    _write_json(summary, batch_dir / "batch_summary.json")
    print(json.dumps({"output_root": _path_text(batch_dir), "completed": summary["completed"], "failed": failed}, indent=2))
    return 1 if failed else 0


def _default_manifest_path() -> Path:
    env_path = os.environ.get("PHASE3_MASK_EDIT_MANIFEST")
    if env_path:
        return Path(env_path)
    if REMOTE_DEFAULT_MANIFEST.exists():
        return REMOTE_DEFAULT_MANIFEST
    return REPO_DEFAULT_MANIFEST


def _load_ui_backend():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import scripts.phase3_end_to_end_ui as ui

    return ui


def _prepare_state(ui, case: dict[str, Any], paths: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    image_path = _copy_input(paths["source_image"], inputs_dir / _input_name(paths["source_image"], "source_image.png"))
    tissue_path = _copy_input(paths["source_tissue_mask"], inputs_dir / "source_tissue_mask.png")
    nuclei_path = _copy_input(paths["source_nuclei_mask"], inputs_dir / "source_cell_mask.png")

    image = ui._load_rgb_image(image_path)
    tissue = ui.load_id_mask(tissue_path)
    nuclei = ui._load_uint8_mask(nuclei_path)
    ui._validate_same_size(image, tissue, "source_tissue_mask")
    ui._validate_same_size(image, nuclei, "source_cell_mask")
    stage_paths = ui._save_pre_generation_artifacts(
        output_dir=output_dir,
        reference_image=image,
        reference_tissue=tissue,
        target_tissue=tissue,
        semantic_change_region=ui.np.zeros(tissue.shape, dtype=bool),
        change_region=ui.np.zeros(tissue.shape, dtype=bool),
    )
    return {
        "profile": case.get("profile") or case.get("dataset"),
        "output_dir": _path_text(output_dir),
        "reference_image": _path_text(image_path),
        "reference_tissue_mask": _path_text(tissue_path),
        "reference_nuclei_mask": _path_text(nuclei_path),
        "source_mask_rgb": stage_paths["source_mask_rgb"],
        "target_mask_rgb": stage_paths["source_mask_rgb"],
        "manifest_case_id": case.get("case_id"),
    }


def _run_tissue_stage(
    ui,
    state: dict[str, Any],
    case: dict[str, Any],
    variant: dict[str, Any],
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _is_no_op_case(case) and not args.execute_no_op_cases:
        return _run_no_op_parser_check(ui, state, case, variant, runtime, args)

    parser_cfg = _mapping(runtime.get("parser"))
    contour_cfg = _mapping(runtime.get("contour"))
    edit_mode = str(variant.get("edit_mode") or "prompt")
    prompt_parser = _option(args.prompt_parser, parser_cfg.get("prompt_parser"), "api")
    instruction_parser = _option(args.instruction_parser, parser_cfg.get("instruction_parser"), "api")
    api_base_url = _option(args.api_base_url, parser_cfg.get("api_base_url"), ui.DEFAULT_API_BASE_URL)
    api_key_env = _option(args.api_key_env, parser_cfg.get("api_key_env"), ui.DEFAULT_API_KEY_ENV)
    api_model = _option(args.api_model, parser_cfg.get("api_model"), ui.DEFAULT_API_MODEL)
    contour_api_base_url = _option(
        args.contour_api_base_url,
        contour_cfg.get("api_base_url"),
        api_base_url,
    )
    contour_api_key_env = _option(
        args.contour_api_key_env,
        contour_cfg.get("api_key_env"),
        api_key_env,
    )
    contour_api_model = _option(
        args.contour_api_model,
        contour_cfg.get("api_model"),
        api_model,
    )
    organic_seed = (
        args.organic_seed
        if args.organic_seed is not None
        else int(case.get("organic_seed", runtime.get("defaults", {}).get("organic_seed", 0)))
    )
    state, tissue_log, _, _ = ui.run_tissue_stage(
        state,
        edit_mode,
        str(case.get("old_prompt", "")),
        str(case.get("new_prompt", "")),
        str(case.get("instruction", "")),
        instruction_parser,
        None,
        "",
        None,
        None,
        prompt_parser,
        api_base_url,
        api_key_env,
        api_model,
        api_model,
        _option(args.qwen_model_path, parser_cfg.get("qwen_model_path"), ""),
        _option(args.qwen_device, parser_cfg.get("qwen_device"), ui.DEFAULT_QWEN_DEVICE),
        bool(args.no_few_shot or parser_cfg.get("no_few_shot", False)),
        "",
        "",
        _option(args.contour_provider, contour_cfg.get("provider"), "api-multimodal"),
        contour_api_base_url,
        contour_api_key_env,
        contour_api_model,
        _option(args.contour_api_image_detail, contour_cfg.get("api_image_detail"), "high"),
        args.fixture_file,
        int(_option(args.max_attempts, contour_cfg.get("max_attempts"), 4)),
        int(_option(args.max_regions, contour_cfg.get("max_regions"), 8)),
        int(_option(args.max_points_per_region, contour_cfg.get("max_points_per_region"), 64)),
        organic_seed,
        bool(args.continue_on_phase3_failure),
    )
    return state, _loads_maybe(tissue_log)


def _run_no_op_parser_check(
    ui,
    state: dict[str, Any],
    case: dict[str, Any],
    variant: dict[str, Any],
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parser_cfg = _mapping(runtime.get("parser"))
    edit_mode = str(variant.get("edit_mode") or "prompt")
    output_dir = Path(state["output_dir"])
    reference_tissue = ui.load_id_mask(state["reference_tissue_mask"])

    if edit_mode == "instruction":
        semantic_diff, parser_info = _resolve_no_op_instruction_diff(
            ui=ui,
            instruction=str(case.get("instruction", "")),
            parser=_option(args.instruction_parser, parser_cfg.get("instruction_parser"), "api"),
            api_base_url=_option(args.api_base_url, parser_cfg.get("api_base_url"), ui.DEFAULT_API_BASE_URL),
            api_key_env=_option(args.api_key_env, parser_cfg.get("api_key_env"), ui.DEFAULT_API_KEY_ENV),
            api_model=_option(args.api_model, parser_cfg.get("api_model"), ui.DEFAULT_API_MODEL),
            output_dir=output_dir,
        )
    else:
        semantic_diff, parser_info = ui._resolve_prompt_semantic_diff(
            old_prompt=str(case.get("old_prompt", "")),
            new_prompt=str(case.get("new_prompt", "")),
            parser=_option(args.prompt_parser, parser_cfg.get("prompt_parser"), "api"),
            api_base_url=_option(args.api_base_url, parser_cfg.get("api_base_url"), ui.DEFAULT_API_BASE_URL),
            api_key_env=_option(args.api_key_env, parser_cfg.get("api_key_env"), ui.DEFAULT_API_KEY_ENV),
            api_model=_option(args.api_model, parser_cfg.get("api_model"), ui.DEFAULT_API_MODEL),
            qwen_model_path=_option(args.qwen_model_path, parser_cfg.get("qwen_model_path"), ""),
            qwen_device=_option(args.qwen_device, parser_cfg.get("qwen_device"), ui.DEFAULT_QWEN_DEVICE),
            no_few_shot=bool(args.no_few_shot or parser_cfg.get("no_few_shot", False)),
            output_dir=output_dir,
        )

    plan = ui.plan_edit_intents(
        semantic_diff,
        reference_profile=state["profile"],
        old_mask=reference_tissue,
        old_prompt=str(case.get("old_prompt", "")),
        new_prompt=str(case.get("new_prompt", "")),
    )
    planned_primitives = [intent.primitive for intent in plan.intents]
    if planned_primitives:
        raise RuntimeError(
            "No-op case produced executable intents: "
            + ", ".join(planned_primitives)
        )

    return _write_no_op_tissue_stage(
        ui,
        state,
        case,
        variant,
        parser_info=parser_info,
        semantic_diff=semantic_diff,
        plan=plan.to_metadata(),
    )


def _resolve_no_op_instruction_diff(
    *,
    ui,
    instruction: str,
    parser: str,
    api_base_url: str,
    api_key_env: str,
    api_model: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return ui._resolve_instruction_semantic_diff(
            instruction=instruction,
            parser=parser,
            api_base_url=api_base_url,
            api_key_env=api_key_env,
            api_model=api_model,
            output_dir=output_dir,
        )
    except Exception as exc:
        if parser == "rule-based" and "Could not infer a supported edit" in str(exc):
            from phase3_mask_edit.parser.semantic_diff import DEFAULT_SEMANTIC_DIFF

            semantic_diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
            return semantic_diff, {
                "mode": "instruction_rule_based",
                "status": "no_supported_edit_inferred",
            }
        raise


def _write_no_op_tissue_stage(
    ui,
    state: dict[str, Any],
    case: dict[str, Any],
    variant: dict[str, Any],
    *,
    parser_info: dict[str, Any] | None = None,
    semantic_diff: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = Path(state["output_dir"])
    reference_image = ui._load_rgb_image(state["reference_image"])
    reference_tissue = ui.load_id_mask(state["reference_tissue_mask"])
    target_path = ui.save_id_mask(reference_tissue, output_dir / "target_mask.png")
    change_region = ui.np.zeros(reference_tissue.shape, dtype=bool)
    stage_paths = ui._save_pre_generation_artifacts(
        output_dir=output_dir,
        reference_image=reference_image,
        reference_tissue=reference_tissue,
        target_tissue=reference_tissue,
        semantic_change_region=change_region,
        change_region=change_region,
    )
    phase3_info = {
        "mode": variant.get("edit_mode"),
        "status": "no_op_manifest_case",
        "primitive": "no_op",
        "expected_primitives": list(case.get("expected_primitives", [])),
        "projection_mode": "no_op",
        "parser": parser_info or {"mode": "not_run"},
        "semantic_diff": semantic_diff,
        "plan": plan,
    }
    ui.save_metadata(phase3_info, output_dir / "phase3_mask_edit" / "execution_summary.json")
    state.update(
        {
            "target_tissue_mask": _path_text(target_path),
            "target_mask_rgb": stage_paths["target_mask_rgb"],
            "change_region": stage_paths["change_region"],
            "phase3": phase3_info,
        }
    )
    info = {
        "status": "tissue_done",
        "edit_mode": variant.get("edit_mode"),
        "primitive": "no_op",
        "changed_area_fraction": 0.0,
        "target_tissue_mask": _path_text(target_path),
        "change_region": stage_paths["change_region"],
    }
    return state, info


def _run_cell_stage(
    ui,
    state: dict[str, Any],
    case: dict[str, Any],
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell_cfg = _mapping(runtime.get("cell"))
    model_paths = _mapping(runtime.get("model_paths"))
    profile = str(case.get("profile") or case.get("dataset") or state.get("profile", "BCSS"))
    dataset = str(case.get("dataset") or profile)
    cell_fill_mode = _option(args.cell_fill_mode, cell_cfg.get("cell_fill_mode"), "preserve")
    probnet_ckpt = _resolve_model_path(
        model_paths,
        "probnet_ckpt",
        args.probnet_ckpt,
        profile,
        dataset,
    )
    nuclei_library = _resolve_model_path(
        model_paths,
        "nuclei_library_template",
        args.nuclei_library_template,
        profile,
        dataset,
    )
    density_scale_json = _resolve_model_path(
        model_paths,
        "density_scale_json_template",
        args.density_scale_json_template,
        profile,
        dataset,
    )
    _validate_probnet_inputs_if_needed(
        cell_fill_mode=cell_fill_mode,
        probnet_ckpt=probnet_ckpt,
        nuclei_library=nuclei_library,
        density_scale_json=density_scale_json,
    )
    state, cell_log, *_ = ui.run_cell_stage(
        state,
        cell_fill_mode,
        _option(args.crossing_cell_policy, cell_cfg.get("crossing_cell_policy"), "delete"),
        probnet_ckpt,
        nuclei_library,
        density_scale_json,
        _option(args.probnet_device, cell_cfg.get("probnet_device"), "auto"),
        str(_option(args.probnet_gamma_values, cell_cfg.get("probnet_gamma_values"), "1.0")),
    )
    return state, _loads_maybe(cell_log)


def _run_generation_stage(
    ui,
    state: dict[str, Any],
    case: dict[str, Any],
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generation_cfg = _mapping(runtime.get("generation"))
    model_paths = _mapping(runtime.get("model_paths"))
    profile = str(case.get("profile") or case.get("dataset") or state.get("profile", "BCSS"))
    state, generation_log, _, _ = ui.run_generation_stage(
        state,
        _option(args.generation_mode, generation_cfg.get("generation_mode"), "dry-run"),
        _option(args.cross_backend, generation_cfg.get("cross_backend"), "cross-v1"),
        float(_option(args.route_threshold, generation_cfg.get("route_threshold"), 0.35)),
        _format_profile_path(
            _option(
                args.pretrained_model_name_or_path,
                model_paths.get("pretrained_model_name_or_path"),
                "",
            ),
            profile,
        ),
        _format_profile_path(_option(args.inpaint_checkpoint, model_paths.get("inpaint_checkpoint"), ""), profile),
        _format_profile_path(_option(args.cross_v1_checkpoint, model_paths.get("cross_v1_checkpoint"), ""), profile),
        _format_profile_path(_option(args.pix2pix_checkpoint, model_paths.get("pix2pix_checkpoint"), ""), profile),
        _option(args.device, generation_cfg.get("device"), "cuda"),
    )
    return state, _loads_maybe(generation_log)


def _resolve_case_paths(
    case: dict[str, Any],
    dataset_roots: dict[str, Path],
    *,
    require_exists: bool = False,
) -> dict[str, Path]:
    return {
        "source_image": _resolve_case_path(
            case,
            "source_image",
            dataset_roots,
            require_exists=require_exists,
        ),
        "source_tissue_mask": _resolve_case_path(
            case,
            "source_tissue_mask",
            dataset_roots,
            require_exists=require_exists,
        ),
        "source_nuclei_mask": _resolve_case_path(
            case,
            "source_nuclei_mask",
            dataset_roots,
            require_exists=require_exists,
        ),
    }


def _resolve_case_path(
    case: dict[str, Any],
    field: str,
    dataset_roots: dict[str, Path],
    *,
    require_exists: bool = False,
) -> Path:
    candidates = _case_path_candidates(case, field, dataset_roots)
    if not require_exists:
        return candidates[0]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(_missing_input_message(case, field, candidates))


def _case_path_candidates(
    case: dict[str, Any],
    field: str,
    dataset_roots: dict[str, Path],
) -> list[Path]:
    dataset = str(case.get("dataset", ""))
    root = dataset_roots.get(dataset)
    raw = str(case.get(field) or "")
    rel = str(case.get(f"{field}_relative") or "")
    basename = _case_path_basename(raw, rel)
    candidates: list[Path] = []

    if rel:
        if root is None:
            raise KeyError(f"No data root configured for dataset {dataset!r}")
        candidates.extend(
            _join_relative_candidates_for_dataset(root, dataset, rel)
        )

    if raw:
        if _looks_windows_absolute(raw):
            derived = _derive_relative_from_path(raw, dataset)
            if derived and root is not None:
                candidates.extend(
                    _join_relative_candidates_for_dataset(root, dataset, derived)
                )
            candidates.append(Path(PureWindowsPath(raw)))
        elif _looks_posix_absolute(raw):
            derived = _derive_relative_from_path(raw, dataset)
            if derived and root is not None:
                candidates.extend(
                    _join_relative_candidates_for_dataset(root, dataset, derived)
                )
            candidates.append(Path(PurePosixPath(raw)))
        else:
            path = Path(raw)
            if path.is_absolute():
                candidates.append(path)
            elif root is not None:
                candidates.extend(
                    _join_relative_candidates_for_dataset(root, dataset, raw)
                )
            else:
                raise KeyError(
                    f"No data root configured for relative {field} in dataset {dataset!r}"
                )
    elif not rel:
        raise KeyError(f"{case.get('case_id', '<case>')} missing {field}")

    if root is not None and basename:
        patch_dir = PATCH_DIR_BY_DATASET.get(dataset)
        for subdir in FIELD_SUBDIRS.get(field, ()):
            candidates.append(_join_manifest_path(root, f"{subdir}/{basename}"))
            if patch_dir:
                candidates.append(_join_manifest_path(root, f"{patch_dir}/{subdir}/{basename}"))
                if _path_name_matches(root, dataset):
                    candidates.append(
                        _join_manifest_path(
                            root.parent,
                            f"{patch_dir}/{subdir}/{basename}",
                        )
                    )

    return _dedupe_paths(candidates)


def _join_relative_candidates_for_dataset(
    root: Path,
    dataset: str,
    relative: str,
) -> list[Path]:
    candidates = [_join_manifest_path(root, relative)]
    patch_dir = PATCH_DIR_BY_DATASET.get(dataset)
    if patch_dir and _relative_starts_with(relative, patch_dir) and _path_name_matches(root, dataset):
        candidates.append(_join_manifest_path(root.parent, relative))
    return candidates


def _relative_starts_with(relative: str, dirname: str) -> bool:
    first = PurePosixPath(relative.replace("\\", "/")).parts[:1]
    return bool(first) and first[0].lower() == dirname.lower()


def _path_name_matches(path: Path, name: str) -> bool:
    return PurePosixPath(str(path).replace("\\", "/")).name.lower() == name.lower()


def _case_path_basename(raw: str, rel: str) -> str:
    source = rel or raw
    if not source:
        return ""
    if _looks_windows_absolute(source):
        return PureWindowsPath(source).name
    return PurePosixPath(source.replace("\\", "/")).name


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = _path_text(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _missing_input_message(case: dict[str, Any], field: str, candidates: list[Path]) -> str:
    case_id = case.get("case_id", "<case>")
    dataset = case.get("dataset", "<dataset>")
    tried = "\n".join(f"  - {_path_text(path)}" for path in candidates)
    return (
        f"Input file does not exist for case_id={case_id}, dataset={dataset}, "
        f"field={field}.\n"
        f"Tried:\n{tried}\n"
        "If this case belongs to a different dataset, edit its `dataset` and "
        "`source_*_relative` fields in the manifest. For your current layout, "
        "runtime.data_roots entries should usually be `/data/wqx/flowedit/data`, "
        "the directory that directly contains BCSS_PATCHES, GlaS_PATCHES, etc."
    )


def _derive_relative_from_path(raw: str, dataset: str) -> str | None:
    patch_dir = PATCH_DIR_BY_DATASET.get(dataset)
    if not patch_dir:
        return None
    for cls in (PureWindowsPath, PurePosixPath):
        parts = cls(raw).parts
        lower_parts = [part.lower() for part in parts]
        try:
            index = lower_parts.index(patch_dir.lower())
        except ValueError:
            continue
        return "/".join(parts[index:])
    return None


def _dataset_roots(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    runtime = _mapping(manifest.get("runtime"))
    roots = {
        str(key): _manifest_path(str(value))
        for key, value in _mapping(runtime.get("data_roots")).items()
        if value
    }
    if args.data_root is not None:
        for dataset in manifest.get("datasets", []):
            roots[str(dataset)] = _manifest_path(args.data_root)
    for key, value in _parse_key_value(args.dataset_root, "--dataset-root").items():
        roots[key] = _manifest_path(value)
    return roots


def _selected_variants(runtime: dict[str, Any], variants_csv: str | None) -> list[dict[str, Any]]:
    raw_variants = runtime.get("edit_variants") or ["prompt", "instruction"]
    variants: list[dict[str, Any]] = []
    for raw in raw_variants:
        if isinstance(raw, str):
            variants.append({"variant_id": raw, "edit_mode": raw})
        elif isinstance(raw, dict):
            edit_mode = raw.get("edit_mode") or raw.get("variant_id")
            if edit_mode:
                variants.append({"variant_id": raw.get("variant_id", edit_mode), **raw, "edit_mode": edit_mode})
    if variants_csv:
        wanted = {part.strip() for part in variants_csv.split(",") if part.strip()}
        variants = [
            variant
            for variant in variants
            if variant.get("variant_id") in wanted or variant.get("edit_mode") in wanted
        ]
    if not variants:
        raise SystemExit("No edit variants selected.")
    return variants


def _selected_cases(manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = list(manifest.get("cases", []))
    if args.dataset:
        wanted = set(args.dataset)
        cases = [case for case in cases if case.get("dataset") in wanted]
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.get("case_id") in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No manifest cases selected.")
    return cases


def _copy_input(source: Path, target: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _input_name(path: Path, fallback: str) -> str:
    suffix = path.suffix.lower()
    return f"source_image{suffix}" if suffix else fallback


def _is_no_op_case(case: dict[str, Any]) -> bool:
    return str(case.get("primitive", "")).lower() == "no_op" or case.get("expected_primitives") == []


def _option(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return ""


def _format_profile_path(value: Any, profile: str) -> str:
    text = str(value or "")
    if not text:
        return ""
    return text.format(profile=profile, profile_lower=profile.lower())


def _resolve_model_path(
    model_paths: dict[str, Any],
    key: str,
    override: Any,
    profile: str,
    dataset: str,
) -> str:
    if override:
        return _normalize_model_path_key(
            key,
            _format_profile_path(override, profile),
            profile,
        )

    by_dataset_keys = []
    if key.endswith("_template"):
        by_dataset_keys.append(f"{key.removesuffix('_template')}_by_dataset")
    by_dataset_keys.append(f"{key}_by_dataset")

    for mapping_key in by_dataset_keys:
        by_dataset = _mapping(model_paths.get(mapping_key))
        for lookup_key in _profile_lookup_keys(profile, dataset):
            value = by_dataset.get(lookup_key)
            if value:
                return _normalize_model_path_key(
                    key,
                    _format_profile_path(value, profile),
                    profile,
                )

    return _normalize_model_path_key(
        key,
        _format_profile_path(model_paths.get(key, ""), profile),
        profile,
    )


def _normalize_model_path_key(key: str, value: str, profile: str) -> str:
    if not value:
        return ""
    path = value.rstrip("/\\")
    if key == "nuclei_library_template" and "{" not in value:
        if _has_statistics_json(path):
            return path
        dataset_path = f"{path}/{profile}"
        if _has_statistics_json(dataset_path):
            return dataset_path
        return path
    if key == "density_scale_json_template" and not path.lower().endswith(".json"):
        return f"{path}/density_scale_{profile.lower()}.json"
    return value


def _validate_probnet_inputs_if_needed(
    *,
    cell_fill_mode: str,
    probnet_ckpt: str,
    nuclei_library: str,
    density_scale_json: str,
) -> None:
    if cell_fill_mode != "probnet":
        return
    missing: list[str] = []
    if probnet_ckpt and not Path(probnet_ckpt).exists():
        missing.append(f"probnet_ckpt missing: {probnet_ckpt}")
    if nuclei_library and not _has_statistics_json(nuclei_library):
        missing.append(f"nuclei_library missing statistics.json: {nuclei_library}")
    if density_scale_json and not Path(density_scale_json).exists():
        missing.append(f"density_scale_json missing: {density_scale_json}")
    if missing:
        detail = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "ProbNet input validation failed before running generate.py:\n"
            f"{detail}\n"
            "Set runtime.model_paths.nuclei_library_template to the exact "
            "directory containing statistics.json, or use "
            "nuclei_library_by_dataset for per-dataset library directories."
        )


def _has_statistics_json(path: str | Path) -> bool:
    return (Path(path) / "statistics.json").exists()


def _profile_lookup_keys(profile: str, dataset: str) -> tuple[str, ...]:
    values = [
        dataset,
        profile,
        dataset.upper(),
        profile.upper(),
        dataset.lower(),
        profile.lower(),
    ]
    return tuple(dict.fromkeys(value for value in values if value))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _loads_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return payload if isinstance(payload, dict) else {"value": payload}
    return {}


def _stringify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stringify(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_stringify(item) for item in value]
    if isinstance(value, Path):
        return _path_text(value)
    return value


def _looks_windows_absolute(value: str) -> bool:
    return len(value) >= 3 and value[1:3] in {":\\", ":/"} and value[0].isalpha()


def _looks_posix_absolute(value: str) -> bool:
    return value.startswith("/") and not _looks_windows_absolute(value)


def _manifest_path(value: str | Path) -> Path:
    text = str(value)
    if _looks_posix_absolute(text):
        return Path(PurePosixPath(text))
    return Path(text)


def _join_manifest_path(root: Path, relative: str) -> Path:
    if _looks_posix_absolute(str(root)):
        parts = PurePosixPath(relative).parts
        return Path(PurePosixPath(str(root), *parts))
    return root / Path(relative)


def _path_text(path: Path) -> str:
    text = str(path)
    if text.startswith("\\") and len(text) > 1:
        candidate = "/" + text.lstrip("\\").replace("\\", "/")
        if any(candidate.startswith(anchor) for anchor in POSIX_ROOT_ANCHORS):
            return candidate
    return path.as_posix()


def _parse_key_value(items: list[str] | None, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{label} expects DATASET=/path, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f"{label} expects DATASET=/path, got {item!r}")
        parsed[key] = value
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_stringify(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 mask-edit manifest cases through the UI organic_v2 backend.",
    )
    parser.add_argument("--manifest", type=Path, default=_default_manifest_path())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", help="Optional output subdirectory name; defaults to a timestamp.")
    parser.add_argument("--plan-only", action="store_true", help="Write and print the expanded batch plan only.")
    parser.add_argument("--variants", help="Comma-separated variant ids/edit modes, e.g. prompt,instruction.")
    parser.add_argument("--case-id", action="append", help="Run only this case id; may be repeated.")
    parser.add_argument("--dataset", action="append", help="Run only this dataset; may be repeated.")
    parser.add_argument("--limit", type=int, help="Limit cases after filters.")
    parser.add_argument("--data-root", type=Path, help="Override all dataset roots as DATA_ROOT/<dataset>.")
    parser.add_argument("--dataset-root", action="append", help="Override one dataset root, e.g. GlaS=/data/.../GlaS.")
    parser.add_argument("--stop-after", choices=("tissue", "cell", "generation"), default="generation")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--continue-on-phase3-failure", action="store_true")
    parser.add_argument("--execute-no-op-cases", action="store_true", help="Send no-op cases through parser/contour instead of writing zero-change outputs.")

    parser.add_argument("--prompt-parser", choices=("api", "qwen-local"))
    parser.add_argument("--instruction-parser", choices=("api", "rule-based"))
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--api-model")
    parser.add_argument("--qwen-model-path")
    parser.add_argument("--qwen-device")
    parser.add_argument("--no-few-shot", action="store_true")

    parser.add_argument("--contour-provider", choices=("api-text", "api-multimodal", "fixture"))
    parser.add_argument("--contour-api-base-url")
    parser.add_argument("--contour-api-key-env")
    parser.add_argument("--contour-api-model")
    parser.add_argument("--contour-api-image-detail", choices=("low", "high", "auto"))
    parser.add_argument("--fixture-file", type=Path)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--max-regions", type=int)
    parser.add_argument("--max-points-per-region", type=int)
    parser.add_argument("--organic-seed", type=int)

    parser.add_argument("--cell-fill-mode", choices=("preserve", "blank", "probnet"))
    parser.add_argument("--crossing-cell-policy", choices=("delete", "keep", "majority"))
    parser.add_argument("--probnet-ckpt")
    parser.add_argument("--nuclei-library-template")
    parser.add_argument("--density-scale-json-template")
    parser.add_argument("--probnet-device")
    parser.add_argument("--probnet-gamma-values")

    parser.add_argument("--generation-mode", choices=("agentic", "dry-run", "auto", "inpaint", "cross-v1"))
    parser.add_argument("--cross-backend", choices=("cross-v1",))
    parser.add_argument("--route-threshold", type=float)
    parser.add_argument("--device")
    parser.add_argument("--pretrained-model-name-or-path")
    parser.add_argument("--inpaint-checkpoint")
    parser.add_argument("--cross-v1-checkpoint")
    parser.add_argument("--pix2pix-checkpoint")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
