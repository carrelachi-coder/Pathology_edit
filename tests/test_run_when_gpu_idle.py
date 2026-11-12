from scripts.run_when_gpu_idle import parse_rows


def test_parse_rows_filters_on_memory_and_utilization() -> None:
    rows = "0, 1000, 0\n1, 21000, 0\n2, 1000, 11\n"
    assert parse_rows(rows, maximum_utilization=10, maximum_memory_mib=20000) == [0]


def test_parse_rows_rejects_malformed_rows() -> None:
    try:
        parse_rows("0, 1", maximum_utilization=10, maximum_memory_mib=20000)
    except ValueError as error:
        assert "unexpected nvidia-smi row" in str(error)
    else:
        raise AssertionError("malformed row was accepted")
