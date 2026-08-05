---
name: breast-discohesive-single-file
description: Plan and audit paired tissue and nuclei edits for discohesive or single-file breast carcinoma invasion. Use only when H&E supports sparse cells or short chains infiltrating stroma or adipose without a cohesive bulk front.
---

# Breast discohesive joint editing

1. Read `references/joint_contract.json` and require direct H&E evidence.
2. Preserve the non-tumor tissue mask where invasion is cellular rather than bulk.
3. Use deterministic single/short-cord layouts; let ProbNet rank only legal placements.
4. Reject adipose, vessel, duct, lobule, or background destruction.
5. Audit the paired condition and post-generation H&E together.
