---
name: panda-gleason-v1
description: Interpret masks following the PANDA Radboud-style stroma, benign epithelium, and Gleason 3/4/5 protocol. Use when annotation_profile_id is panda-gleason-v1, regardless of pathology domain metadata.
---

# PANDA annotation protocol

1. Read `references/mask_contract.json` first; enforce provider/remap, zero, fine-ID, and topology constraints. Then read `references/rules.json` for Gleason interpretation.
2. Use this profile only for verified Radboud-style fine masks. Never apply its 0–5 meanings to Karolinska-style coarse masks.
3. Treat label zero as immutable background/unknown/unannotated support. Never seed, bridge, fill, or expand through it.
4. Preserve Gleason 3/4/5 fine IDs during burden-only edits. State explicitly that mask ID 9 cannot guarantee fused/cribriform/poorly formed architecture.
5. Hand pattern rendering to generation/post-generation audit; cite constraints and rules and abstain when provider, fine-mask identity, or required gland/lumen map is uncertain.
