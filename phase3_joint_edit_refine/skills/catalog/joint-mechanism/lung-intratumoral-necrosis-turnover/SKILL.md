---
name: lung-intratumoral-necrosis-turnover
description: Plan and audit interface-bound lung tumor necrosis appearance or resolution while preserving airway and alveolar spaces. Use for lung necrosis primitives.
---

# Lung intratumoral necrosis turnover

Read `references/joint_contract.json`. Require a real viable-tumor/necrosis interface. For appearance, replace the changed-core population with a sparse ProbNet population restricted to inflammatory/dead CellViT classes; delegate only non-nuclear debris and karyorrhectic texture to bounded H&E rendering. For resolution, remove observed dead instances and regenerate viable tumor nuclei. Preserve airway lumen, alveolar airspace, fluid, cartilage, vessels and papillary cores. Never treat mucus, a cavity or an airspace as necrosis, and reject dense uniform dead-nucleus filling.
