import copy

import pytest

from scripts.analyze_manuscript_formal_r3 import analyze, validate_report


def measurement(value, status="valid_nonzero"):
    return {"status": status, "value": value}


def formal_report(n=4):
    records = []
    for index in range(n):
        values = {
            "original": (0.10, 0.20, 1.0),
            "sham": (0.11, 0.21, 2.0),
            "u1_moderate": (0.20, 0.21, 2.0),
            "u1_significant": (0.30, 0.21, 2.0),
            "u2_moderate": (0.11, 0.30, 4.0),
            "u2_significant": (0.11, 0.40, 6.0),
        }
        for state, (u1, u2, cells) in values.items():
            records.append(
                {
                    "patient_id": f"p{index}",
                    "reference_id": f"r{index}",
                    "state": state,
                    "tissue": {"u1": measurement(u1), "u2": measurement(u2)},
                    "cell_status": "valid_nonzero",
                    "inflammatory_density": measurement(cells),
                }
            )
    return {
        "scope": "formal_r3",
        "formal_launch_permitted": True,
        "status": "complete",
        "chain_integrity_accepted": True,
        "records": records,
    }


def test_complete_series_produces_four_holm_comparisons_and_orders():
    result = analyze(formal_report(), seed=7, bootstrap_draws=100, permutation_draws=100)
    assert result["requested_patients"] == 4
    assert len(result["primary_holm_family"]) == 4
    assert all(row["valid_pairs"] == 4 for row in result["primary_holm_family"])
    assert all("holm_adjusted_p_value" in row for row in result["primary_holm_family"])
    assert all(row["strict_order"] == 4 for row in result["order_outcomes"])


def test_valid_zero_is_retained_but_uncertain_is_not_imputed():
    report = formal_report()
    report["records"][1]["tissue"]["u1"] = measurement(0, "valid_zero_with_completion")
    report["records"][7]["tissue"]["u1"] = measurement(None, "evaluator_uncertain")
    result = analyze(report, seed=7, bootstrap_draws=20, permutation_draws=20)
    u1_moderate = next(
        row for row in result["primary_holm_family"] if row["edit_state"] == "u1_moderate"
    )
    assert u1_moderate["valid_pairs"] == 3
    assert sum(u1_moderate["excluded_pairs_by_status"].values()) == 1


def test_order_reports_ties_reversals_and_all_requested_bounds():
    report = formal_report(3)
    by_patient_state = {(row["patient_id"], row["state"]): row for row in report["records"]}
    by_patient_state[("p0", "u1_moderate")]["tissue"]["u1"] = measurement(0.11)
    by_patient_state[("p1", "u1_significant")]["tissue"]["u1"] = measurement(0.05)
    by_patient_state[("p2", "u1_significant")]["tissue"]["u1"] = measurement(None, "missing")
    result = analyze(report, seed=7, bootstrap_draws=20, permutation_draws=20)
    u1 = next(row for row in result["order_outcomes"] if row["endpoint"] == "u1")
    assert (u1["adjacent_tie"], u1["reversal"], u1["incomplete"]) == (1, 1, 1)
    assert u1["strict_order_all_requested_lower_bound"] == 0
    assert u1["strict_order_all_requested_upper_bound"] == pytest.approx(1 / 3)


def test_technical_or_incomplete_report_is_rejected():
    report = formal_report()
    report["scope"] = "joint_gpu_technical_acceptance"
    with pytest.raises(ValueError, match="not a formal"):
        validate_report(report)
    report = formal_report()
    report["chain_integrity_accepted"] = False
    with pytest.raises(ValueError, match="incomplete"):
        validate_report(report)


def test_multiple_sources_for_one_patient_are_rejected():
    report = formal_report()
    report["records"][0] = copy.deepcopy(report["records"][0])
    report["records"][0]["reference_id"] = "different"
    with pytest.raises(ValueError, match="one source"):
        validate_report(report)
