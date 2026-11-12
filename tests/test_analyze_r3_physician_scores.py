import csv
import json
from pathlib import Path

from scripts.analyze_r3_physician_scores import analyze_order, analyze_realism


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_realism_pairs_within_patient_and_reader(tmp_path: Path):
    records = []
    models = [
        ("Ours global recon", "o"),
        ("Real H&E", "r"),
        ("PixCell", "p"),
        ("PathDiff mask", "d"),
        ("MuPaD text", "t"),
        ("MuPaD image", "i"),
        ("PathDiff text", "x"),
        ("UniPath", "u"),
    ]
    for case in range(60):
        for model, suffix in models:
            records.append(
                {
                    "blind_id": f"{case:02d}{suffix}",
                    "case": f"C{case:03d}",
                    "model_label": model,
                    "organ": "breast",
                }
            )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"records": records}))
    score_paths = []
    for reader in (1, 2):
        rows = []
        for record in records:
            rows.append(
                {
                    "医生": reader,
                    "匿名编号": record["blind_id"],
                    "真实性评分": 4 if record["model_label"].startswith("Ours") else 5,
                    "无法判断": "",
                }
            )
        path = tmp_path / f"reader{reader}.csv"
        write_csv(path, rows)
        score_paths.append(path)
    result = analyze_realism(
        mapping,
        score_paths,
        bootstrap_draws=100,
        permutation_draws=100,
        seed=3,
    )
    comparison = next(
        row
        for row in result["ours_paired_comparisons"]
        if row["comparison_model"] == "Real H&E"
    )
    assert comparison["valid_patients"] == 60
    assert comparison["mean_paired_difference"] == -1
    assert result["doctor_scores_complete"] is True


def test_order_uses_sham_moderate_significant_and_keeps_original_secondary(tmp_path: Path):
    key_rows = []
    score_paths = []
    for reader in (1, 2):
        score_rows = []
        for operation in ("u1", "u2"):
            group = f"R{reader}_{operation}"
            states = {"A": "original", "B": "sham", "C": "moderate", "D": "significant"}
            for letter, state in states.items():
                key_rows.append(
                    {
                        "reader": reader,
                        "group_id": group,
                        "display_letter": letter,
                        "patient_id": "P1",
                        "operation": operation,
                        "state": state,
                    }
                )
            score_rows.append(
                {
                    "组编号": group,
                    "A分值": 2,
                    "B分值": 1,
                    "C分值": 2,
                    "D分值": 3,
                    "无法判断": "",
                }
            )
        path = tmp_path / f"order{reader}.csv"
        write_csv(path, score_rows)
        score_paths.append(path)
    key = tmp_path / "key.csv"
    write_csv(key, key_rows)
    result = analyze_order(key, score_paths)
    assert all(row["strict_edit_order"] for row in result["group_records"])
    assert not any(row["original_sham_tie"] for row in result["group_records"])
    assert result["qualification_updated"] is False
