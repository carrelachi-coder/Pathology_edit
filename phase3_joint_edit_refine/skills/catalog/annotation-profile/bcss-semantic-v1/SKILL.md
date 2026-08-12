---
name: bcss-semantic-v1-joint
description: Compose joint tissue-nucleus edits under the BCSS semantic profile. Use for breast or cross-domain patches annotated with bcss-semantic-v1.
---

# BCSS joint annotation profile

1. Read the tissue profile at `../../../../../phase3_mask_edit_refine/skills/catalog/annotation-profile/bcss-semantic-v1/SKILL.md`.
2. Read `references/joint_contract.json` before exposing mechanisms.
3. Resolve native fine labels before using canonical vessel, necrosis, immune aggregate, or benign epithelium.
4. Preserve background and all unrequested fine labels pixel-exactly.
5. Treat unified fine ID 2 as operational Stroma by default. Native BCSS
   stroma and glandular secretions collapse into this unified ID, so neither
   the Planner nor the critic may infer fibrosis, tumor-bed identity, lumen,
   or secretion from H&E. Geometry remains executable under the operational
   label, while all of those histologic claims remain unavailable.
6. Breast invasive edits may use fine ID 1 only. Preserve DCIS 14 and
   angioinvasion 15 unless a separately reviewed primitive explicitly names them.
7. Keep added nuclei compatible with the resolved target fine label.
8. Supply the LLM only with tissue/nuclei masks, graph overlays, skill rules,
   and deterministic candidate certificates. Raw H&E is an audit and render
   input, not an execution-planning authority.
