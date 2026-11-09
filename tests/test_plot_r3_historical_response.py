import copy
import pytest
from scripts.plot_r3_historical_response import extract, METRICS


def report():
    analysis = {
        "sample_count": 300,
        "cluster_count": 90,
        "cluster_unit": "patient_id",
        "primitives": {
            p: {
                "backends": {
                    b: {
                        m: {"count": 90, "mean": 0.5, "ci95_low": 0.4, "ci95_high": 0.6}
                        for m, *_ in METRICS
                    }
                    for b in ["inpaint", "cross"]
                }
            }
            for p in ["u1", "u2"]
        },
    }
    return {
        "status": "complete",
        "schema_version": 2,
        "inference": {
            k: "patient_id"
            for k in ["bootstrap_unit", "permutation_unit", "correlation_unit"]
        },
        "encoders": {
            "uni2h": copy.deepcopy(analysis),
            "conch": copy.deepcopy(analysis),
        },
    }


def test_all_metric_rows_are_read_from_patient_summaries():
    rows = extract(report())
    assert len(rows) == 24 and all(
        r["patients"] == 90 and r["patches"] == 300 for r in rows
    )


@pytest.mark.parametrize(
    "fault", ["old_unit", "incomplete", "patch_count", "nonfinite"]
)
def test_invalid_or_wrong_inference_cannot_generate_result_rows(fault):
    d = report()
    if fault == "old_unit":
        d["inference"]["permutation_unit"] = "wsi_id"
    if fault == "incomplete":
        d["status"] = "running"
    if fault == "patch_count":
        d["encoders"]["uni2h"]["sample_count"] = 299
    if fault == "nonfinite":
        d["encoders"]["conch"]["primitives"]["u2"]["backends"]["cross"][METRICS[0][0]][
            "ci95_low"
        ] = float("nan")
    with pytest.raises(ValueError):
        extract(d)
