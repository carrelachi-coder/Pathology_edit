from pathlib import Path
from argparse import Namespace

import numpy as np
import pytest

from phase3_mask_edit.benchmark.generation_models import (
    load_generation_model_configs,
)
from scripts.normalize_generation_mpp import MODEL_SPECS, normalize_image
from scripts.prepare_mupad_wsi_context import (
    box_from_patch,
    boxes_overlap,
    central_match_is_valid,
    context_box_from_patch,
)
from scripts.resume_generation_after_mupad_context import (
    context_ready,
    prerequisite_queues_complete,
)
from scripts.run_generation_baseline_queue import build_commands
from scripts.run_generation_baseline import (
    nominal_mpp_for_objective,
    pathdiff_text_effective_prompt,
)
from scripts.validate_generation_baselines import validate_normalized

from PIL import Image


CONFIG_DIR = Path("benchmark_configs/models")


def test_generation_model_configs_freeze_expected_contracts():
    configs = load_generation_model_configs(CONFIG_DIR)

    assert set(configs) == {
        "cross_v1_project",
        "pixcell_controlnet",
        "pathdiff_conic",
        "pathdiff_text",
        "pathldm_plip",
        "unipath_7b",
        "mupad_text",
        "mupad_image_auxiliary",
    }
    assert configs["pixcell_controlnet"].allowed_inputs == (
        "reference_image",
        "target_nuclei_mask",
    )
    assert configs["pathdiff_conic"].allowed_inputs == (
        "target_conic_instance_type_mask",
        "prompt",
    )
    assert configs["pathdiff_text"].allowed_inputs == ("prompt",)
    assert "target_conic_instance_type_mask" in configs[
        "pathdiff_text"
    ].forbidden_inputs
    assert configs["mupad_image_auxiliary"].allowed_inputs == (
        "reference_wsi_context",
    )
    for config in configs.values():
        assert "target_image" not in config.allowed_inputs
        assert "target_image" in config.forbidden_inputs


def test_generation_model_config_discovery_ignores_appledouble_files(tmp_path):
    for path in CONFIG_DIR.glob("*.yaml"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    (tmp_path / "._pixcell.yaml").write_bytes(b"\x00\x05\x16\x07invalid")

    configs = load_generation_model_configs(tmp_path)

    assert "pixcell_controlnet" in configs


def test_pathdiff_command_uses_conic_masks_and_is_resumable():
    config = load_generation_model_configs(CONFIG_DIR)["pathdiff_conic"]

    command = config.build_remote_command(
        manifest="/tmp/manifest.json",
        output_root="/tmp/output",
        device="cuda:0",
        max_items=1,
        num_shards=3,
        shard_index=1,
    )

    assert "--model-type pathdiff" in command
    assert "--conic-root" in command
    assert "--max-items 1" in command
    assert "--num-shards 3 --shard-index 1" in command
    assert "target_image" not in command
    assert "--overwrite" not in command


def test_pathdiff_text_command_uses_official_t2i_mode_without_conic_mask():
    config = load_generation_model_configs(CONFIG_DIR)["pathdiff_text"]

    command = config.build_remote_command(
        manifest="/tmp/manifest.json",
        output_root="/tmp/output",
        device="cuda:0",
        max_items=1,
    )

    assert "--model-type pathdiff" in command
    assert "--pathdiff-mode t2i" in command
    assert "--conic-root" not in command
    assert "--native-mpp 0.5" in command
    assert "--pathdiff-text-objective-magnification 20" in command
    assert "--max-items 1" in command


def test_pathdiff_text_scale_prompt_is_prefixed_and_auditable():
    prompt, metadata = pathdiff_text_effective_prompt(
        "Pleomorphic tumor cells with stromal inflammation.", 20
    )

    assert prompt.startswith(
        "H&E-stained histopathology at 20x objective magnification."
    )
    assert prompt.endswith("Pleomorphic tumor cells with stromal inflammation.")
    assert metadata == {
        "kind": "prompt_conditioned_objective_magnification",
        "objective_magnification": 20.0,
        "nominal_mpp": 0.5,
        "prompt_position": "prefix",
        "scale_prompt": (
            "H&E-stained histopathology at 20x objective magnification."
        ),
        "physical_scale_status": "prompt_conditioned_nominal_scale",
    }
    assert nominal_mpp_for_objective(40) == 0.25


def test_pathdiff_text_scale_prompt_rejects_conflicting_source_scale():
    with pytest.raises(ValueError, match="already contains a scale term"):
        pathdiff_text_effective_prompt(
            "H&E-stained tissue at 40x magnification.", 20
        )


def test_pathldm_command_includes_training_condition_levels():
    config = load_generation_model_configs(CONFIG_DIR)["pathldm_plip"]

    command = config.build_remote_command(
        manifest="/tmp/manifest.json",
        output_root="/tmp/output",
        device="cuda:0",
    )

    assert "--pathldm-tumor-level high" in command
    assert "--pathldm-til-level low" in command


def test_mupad_image_command_requires_real_wsi_context_and_overlap_exclusions():
    config = load_generation_model_configs(CONFIG_DIR)["mupad_image_auxiliary"]

    command = config.build_remote_command(
        manifest="/tmp/manifest.json",
        output_root="/tmp/output",
        device="cuda:0",
    )

    assert "--mupad-context-root" in command
    assert "--exclude-sample-ids" in command
    assert "reflect" not in command


def test_mupad_wsi_context_is_centered_on_the_exact_reference_patch():
    reference = {"x": "57728", "y": "33696"}
    target_far = {"x": "66464", "y": "58560"}
    target_near = {"x": "58400", "y": "33696"}

    patch_box = box_from_patch(reference, patch_size=512, inner_offset=80)
    context_box = context_box_from_patch(
        reference, patch_size=512, context_size=1024, inner_offset=80
    )

    assert patch_box == [57808, 33776, 58320, 34288]
    assert context_box == [57552, 33520, 58576, 34544]
    assert not boxes_overlap(
        context_box, box_from_patch(target_far, patch_size=512, inner_offset=80)
    )
    assert boxes_overlap(
        context_box, box_from_patch(target_near, patch_size=512, inner_offset=80)
    )


def test_mupad_context_resume_requires_complete_valid_common_subset():
    summary = {
        "missing_wsi_count": 0,
        "missing_context_records": [],
        "failures": [],
        "completed_this_run": 100,
        "skipped_complete": 1367,
        "eligible_direction_count": 1467,
    }

    assert context_ready(summary)
    summary["missing_wsi_count"] = 1
    assert not context_ready(summary)


def test_mupad_central_reference_allows_only_negligible_decoder_rounding():
    exact = np.zeros((4, 4, 3), dtype=np.int16)
    one_level = exact.copy()
    one_level[0, 0, 0] = 1
    two_levels = exact.copy()
    two_levels[0, 0, 0] = 2

    assert central_match_is_valid(exact, max_absolute_error=1, max_mae=1e-4)
    assert central_match_is_valid(one_level, max_absolute_error=1, max_mae=0.1)
    assert not central_match_is_valid(two_levels, max_absolute_error=1, max_mae=0.1)


def test_mupad_context_resume_waits_for_prerequisite_queues(tmp_path):
    assert prerequisite_queues_complete(None, 2)
    assert not prerequisite_queues_complete(tmp_path, 2)

    for shard_index in range(2):
        (tmp_path / f"queue_shard{shard_index}of2.json").write_text(
            '{"status": "completed"}', encoding="utf-8"
        )
    assert prerequisite_queues_complete(tmp_path, 2)

    (tmp_path / "queue_shard1of2.json").write_text(
        '{"status": "failed"}', encoding="utf-8"
    )
    assert not prerequisite_queues_complete(tmp_path, 2)


def test_cross_v1_reuses_existing_preview_run():
    config = load_generation_model_configs(CONFIG_DIR)["cross_v1_project"]

    assert config.is_reused
    assert "preview_60_20260714" in config.execution["output_root"]


def test_model_configs_pass_native_mpp_and_resolution():
    configs = load_generation_model_configs(CONFIG_DIR)

    for model_id, config in configs.items():
        if config.is_reused:
            continue
        arguments = config.execution["arguments"]
        native_mpp = MODEL_SPECS[model_id]["native_mpp"]
        if native_mpp is None:
            assert "native-mpp" not in arguments
        else:
            assert float(arguments["native-mpp"]) == native_mpp
        assert int(arguments["native-resolution"]) == MODEL_SPECS[model_id][
            "native_resolution"
        ]


def test_unipath_uses_the_official_inference_step_default():
    config = load_generation_model_configs(CONFIG_DIR)["unipath_7b"]

    assert config.execution["arguments"]["steps"] == 30


def test_mpp_normalization_preserves_or_crops_expected_fov():
    cases = {
        "pixcell_controlnet": [0, 0, 256, 256],
        "pathdiff_conic": [0, 0, 256, 256],
        "pathdiff_text": [0, 0, 256, 256],
        "pathldm_plip": [64, 64, 192, 192],
        "unipath_7b": [64, 64, 320, 320],
        "mupad_text": [128, 128, 384, 384],
    }
    for model_id, expected_box in cases.items():
        spec = MODEL_SPECS[model_id]
        image = Image.new(
            "RGB", (spec["native_resolution"], spec["native_resolution"]), "white"
        )
        normalized, box = normalize_image(
            image,
            spec["native_mpp"],
            512,
            0.25,
            spec.get("normalization_strategy", "physical_fov_center_crop"),
        )
        assert box == expected_box
        assert normalized.size == (512, 512)


def test_normalized_validation_only_checks_selected_models(tmp_path):
    records = [
        {
            "sample_id": "breast-0001-a_to_b",
            "organ": "breast",
        }
    ]
    sample_dir = (
        tmp_path
        / "pixcell_controlnet"
        / "breast"
        / "breast-0001-a_to_b"
    )
    sample_dir.mkdir(parents=True)
    Image.new("RGB", (512, 512), "white").save(sample_dir / "generated.png")
    (sample_dir / "normalization.json").write_text(
        '{"target_mpp": 0.25, "target_fov_um": 128.0, '
        '"center_crop_box_xyxy": [0, 0, 256, 256]}'
    )

    result = validate_normalized(records, tmp_path, ["pixcell_controlnet"])

    assert result == {
        "counts": {"pixcell_controlnet": 1},
        "failures": [],
        "valid": True,
    }


def test_full_generation_queue_excludes_cross_and_unipath():
    args = Namespace(
        config_dir=CONFIG_DIR,
        manifest=Path("/tmp/manifest.json"),
        output_root=Path("/tmp/output"),
        state_root=Path("/tmp/state"),
        models=["pixcell_controlnet", "pathdiff_conic"],
        cuda_visible_device="1",
        num_shards=2,
        shard_index=0,
        attempts=3,
        retry_delay_seconds=60,
        dry_run=True,
    )

    commands = build_commands(args)

    assert [model_id for model_id, _ in commands] == [
        "pixcell_controlnet",
        "pathdiff_conic",
    ]
    assert all("--num-shards 2 --shard-index 0" in command for _, command in commands)

    args.models = ["unipath_7b"]
    with pytest.raises(ValueError, match="explicitly excluded"):
        build_commands(args)
