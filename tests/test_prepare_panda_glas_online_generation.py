from scripts.prepare_panda_glas_online_generation import duplicate_source_groups


def _row(case_id: str, primitive_id: str, digest: str) -> dict[str, object]:
    return {
        "dataset": "GLaS",
        "category": "colorectal-local-population-modulation",
        "primitive_id": primitive_id,
        "case_id": case_id,
        "source_image_sha256": digest,
    }


def test_duplicate_source_groups_rejects_reuse_within_primitive() -> None:
    rows = [_row("case-a", "cellularity-increase-v1", "same")]
    rows.append(_row("case-b", "cellularity-increase-v1", "same"))

    assert list(duplicate_source_groups(rows).values()) == [["case-a", "case-b"]]


def test_duplicate_source_groups_allows_distinct_primitive_edits() -> None:
    rows = [_row("case-a", "cellularity-increase-v1", "same")]
    rows.append(_row("case-b", "cellularity-decrease-v1", "same"))

    assert duplicate_source_groups(rows) == {}
