---
name: puma-semantic-v1
description: Interpret masks following the PUMA melanoma tissue protocol with tumor, stroma, necrosis, epidermis, and blood vessel classes. Use when annotation_profile_id is puma-semantic-v1.
---

# PUMA annotation protocol

1. Read `references/mask_contract.json` first; enforce anatomical provenance, background, tissue-label, nuclei separation, and topology constraints. Then read `references/rules.json` for melanoma interpretation.
2. Interpret canonical `Normal epithelium` strictly as PUMA epidermis and preserve its continuity outside epidermis-specific primitives.
3. Keep semantic tissue and nuclei annotations separate. Never rasterize immune nuclei into an invented immune tissue region.
4. Preserve white background exactly and exclude it from edit sources, anchors, bands, filling, and bridges.
5. Retrieve matched statistics, cite constraints and rules, and hand epidermal/melanoma morphology to generation/post-generation audit. Abstain when the anatomical stratum or raw epithelium mapping is unknown.
