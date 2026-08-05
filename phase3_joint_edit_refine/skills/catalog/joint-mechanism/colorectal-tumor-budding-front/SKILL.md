---
name: colorectal-tumor-budding-front
description: Plan and audit cell-level colorectal tumor budding without inventing detached tumor tissue. Use only when H&E supports isolated neoplastic cells or clusters of up to four cells at a verified invasive front.
---

# Colorectal tumor-budding joint editing

1. Read `references/joint_contract.json` and require high-confidence H&E evidence.
2. Keep the receiving stromal tissue label unchanged for the budding footprint.
3. Generate single, pair, or 3–4 cell templates deterministically; use ProbNet only to rank legal placements.
4. Preserve existing stromal cells, mucin, lumen, muscle, vessels, nerves, and background.
5. Reject bulk fill, bridges between buds, remote cells, and unsupported cell identity.
