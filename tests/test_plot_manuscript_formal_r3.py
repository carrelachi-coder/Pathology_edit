from pathlib import Path

from scripts.analyze_manuscript_formal_r3 import analyze
from scripts.plot_manuscript_formal_r3 import order_count_label, plot
from tests.test_analyze_manuscript_formal_r3 import formal_report


def test_formal_plot_writes_tables_and_two_renderings(tmp_path: Path):
    analysis = analyze(formal_report(), seed=7, bootstrap_draws=20, permutation_draws=20)
    outputs = plot(analysis, tmp_path / "figure")
    assert set(outputs) == {"primary_csv", "order_csv", "sham_drift_csv", "png", "pdf"}
    assert all(Path(item["path"]).is_file() and item["bytes"] > 0 for item in outputs.values())


def test_order_count_label_reports_strict_orders_among_valid_triplets():
    row = {"strict_order": 1, "valid_triplets": 7, "requested_patients": 7}
    assert order_count_label(row) == "1/7"
