#!/usr/bin/env python3
"""Independent regeneration of the pressure fixture, not a production decoder."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def verify(root, count):
    rng = np.random.default_rng(20261103)
    expected_chain = hashlib.sha256()
    with (root / "arrays.index.jsonl").open("rb") as index:
        for occurrence in range(count):
            receipt = json.loads(index.readline())
            expected = rng.random((512, 512), dtype=np.float64).tobytes(order="C")
            expected_sha = hashlib.sha256(expected).hexdigest()
            descriptor = receipt["header"]["descriptor"]
            assert descriptor == {
                "dtype": np.dtype(np.float64).str,
                "shape": [512, 512],
                "content_sha256": expected_sha,
            }
            assert receipt["file"] == f"array_{occurrence:012d}.bin"
            with (root / receipt["file"]).open("rb") as blob:
                assert blob.read(8) == bytes.fromhex("4754524143453200")
                length = struct.unpack(">I", blob.read(4))[0]
                assert length <= 16384
                header = json.loads(blob.read(length))
                assert header == receipt["header"]
                actual = blob.read(len(expected) + 1)
            assert actual == expected, f"snapshot mismatch: {occurrence}"
            expected_chain.update(bytes.fromhex(expected_sha))
        assert not index.readline(), "unexpected additional occurrence"
    return {
        "fixture": "PCG64 seed20261103, 512x512 float64 per occurrence",
        "arrays_regenerated": count,
        "raw_array_bytes_compared": count * 512 * 512 * 8,
        "every_original_payload_byte_equal": True,
        "ordered_reference_sha256_chain": expected_chain.hexdigest(),
        "independent_of_codec_implementation": True,
        "no_pathology_calls": True,
        "numpy_version": np.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", type=Path)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.stream, args.count), indent=2))
