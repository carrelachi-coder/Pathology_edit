from pathlib import Path

from scripts.analyze_manuscript_formal_r3 import analyze
from scripts.plot_manuscript_formal_r3 import plot
from tests.test_analyze_manuscript_formal_r3 import formal_report


def test_formal_plot_writes_tables_and_two_renderings(tmp_path: Path):
    analysis = analyze(formal_report(), seed=7, bootstrap_draws=20, permutation_draws=20)
    outputs = plot(analysis, tmp_path / "figure")
    assert set(outputs) == {"primary_csv", "order_csv", "sham_drift_csv", "png", "pdf"}
    assert all(Path(item["path"]).is_file() and item["bytes"] > 0 for item in outputs.values())
