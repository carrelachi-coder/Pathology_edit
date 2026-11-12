from scripts.measure_manuscript_joint_technical import scope_contract


def test_technical_scope_remains_the_default_contract():
    assert scope_contract(False) == {
        "technical_validation": True,
        "generation_scope": "joint_gpu_technical_acceptance",
        "report_scope": "joint_gpu_technical_acceptance",
        "formal_launch_permitted": False,
    }


def test_formal_scope_requires_formal_generation_and_emits_formal_r3():
    assert scope_contract(True) == {
        "technical_validation": False,
        "generation_scope": "formal_manuscript_cohort",
        "report_scope": "formal_r3",
        "formal_launch_permitted": True,
    }
