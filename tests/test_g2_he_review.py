import json
import tempfile
import unittest
from pathlib import Path

from phase3_joint_edit_refine.g2_he_review import (
    DECISION_STATUSES,
    _decision,
    _prostate_mechanism,
)


class G2HEReviewTests(unittest.TestCase):
    def test_glas_stroma_increase_abstains(self):
        result = _decision(
            case_number=175,
            organ="colorectal",
            legacy_primitive="stromal_desmoplasia",
            record={},
        )
        self.assertEqual(result[0], "abstain")
        self.assertIn("no_stroma_authority", result[4])

    def test_legacy_immune_becomes_cell_only(self):
        result = _decision(
            case_number=241,
            organ="lung",
            legacy_primitive="stromal_immune_infiltration",
            record={},
        )
        self.assertEqual(result[0], "convert_cell_only")
        self.assertEqual(result[1], "cell-type-abundance-increase-v1")
        self.assertEqual(result[2], "lung-local-population-modulation")

    def test_lung_airspace_does_not_seed_necrosis(self):
        result = _decision(
            case_number=272,
            organ="lung",
            legacy_primitive="necrosis_appearance",
            record={},
        )
        self.assertEqual(result[0], "abstain")
        self.assertIn("airspace", result[4])

    def test_sparse_skin_case_uses_infiltration(self):
        result = _decision(
            case_number=599,
            organ="skin",
            legacy_primitive="stroma_decrease",
            record={},
        )
        self.assertEqual(result[0], "replace_primitive")
        self.assertEqual(result[1], "neoplastic-cell-infiltration-increase-v1")

    def test_mixed_panda_case_uses_dominant_native_pattern(self):
        record = {
            "source_statistics": {
                "tissue": {"fine_id_counts": {"2": 100, "8": 20, "9": 80}}
            }
        }
        self.assertEqual(_prostate_mechanism(record), "prostate-pattern-4-growth")

    def test_decision_status_contract_is_closed(self):
        self.assertEqual(
            DECISION_STATUSES,
            {
                "supported_mechanism",
                "rewrite_instruction",
                "convert_cell_only",
                "replace_primitive",
                "abstain",
            },
        )


if __name__ == "__main__":
    unittest.main()
