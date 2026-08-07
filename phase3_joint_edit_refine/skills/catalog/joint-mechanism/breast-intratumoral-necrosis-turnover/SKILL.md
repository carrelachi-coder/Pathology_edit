---
name: breast-intratumoral-necrosis-turnover
description: Plan and audit interface-bound breast tumor necrosis appearance or resolution with viable/dead cell turnover. Use for breast necrosis primitives.
---

# Breast intratumoral necrosis turnover

Read `references/joint_contract.json`. Require an existing annotated intratumoral necrosis focus. For appearance, replace the changed-core population with a sparse ProbNet population restricted to inflammatory/dead CellViT classes; retain H&E rendering responsibility only for non-nuclear necrotic material and karyorrhectic texture. For resolution, remove observed dead instances and regenerate viable tumor nuclei. Expand only Tumor to Necrosis; resolve only Necrosis to verified adjacent Tumor. Preserve duct/lobule lumina, fat and retraction spaces. Reject de-novo foci, residual viable neoplastic nuclei in the new core, dense uniform dead-nucleus filling and unsupported subtype claims.
