import pytest

from scripts.run_reserved_four_target_idle_batch import observe, parse_rows, stable_devices


def test_parse_rows_applies_both_idle_limits() -> None:
    rows = "0, 0, 100\n1, 11, 100\n2, 5, 50001\n3, 10, 50000\n"
    assert parse_rows(rows, maximum_utilization=10, maximum_memory_mib=50000) == [0, 3]


def test_parse_rows_rejects_malformed_or_duplicate_rows() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        parse_rows("0, 1", maximum_utilization=10, maximum_memory_mib=50000)
    with pytest.raises(ValueError, match="duplicate"):
        parse_rows("0, 1, 2\n0, 2, 3", maximum_utilization=10, maximum_memory_mib=50000)


def test_stable_devices_requires_both_observations_and_is_bounded() -> None:
    first = {"eligible_physical_gpu_indices": [0, 1, 3]}
    second = {"eligible_physical_gpu_indices": [1, 2, 3]}
    assert stable_devices(first, second, 1) == [1]
    assert stable_devices(first, second, 8) == [1, 3]


def test_observe_preserves_raw_evidence() -> None:
    row = "2, 4, 123"
    result = observe(lambda: row, maximum_utilization=10, maximum_memory_mib=50000)
    assert result["raw_rows"] == [row]
    assert result["eligible_physical_gpu_indices"] == [2]
