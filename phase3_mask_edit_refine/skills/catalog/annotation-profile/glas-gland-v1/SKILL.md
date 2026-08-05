---
name: glas-gland-v1
description: Interpret masks following the repository's GlaS instance-gland plus patch-grade to unified fine-label preprocessing protocol. Use when annotation_profile_id is glas-gland-v1.
---

# GlaS annotation protocol

1. Read `references/mask_contract.json` first; enforce provenance, zero, label, interface, and conditional native-instance constraints. Then read `references/rules.json` for gland interpretation.
2. Treat gland instances as the authoritative topology. Preserve each gland's boundary and lumen; do not merge, split, or fill glands with semantic morphology.
3. Treat repository fine subtype IDs as field-derived values, not independently annotated per-gland grades.
4. Interpret `Stroma` only as transformed non-gland complement. Use H&E to exclude lumen, mucin, debris, muscle, inflammation, and unrelated tissue before selecting it.
5. Preserve zero exactly, cite constraints and rules, and abstain if the native instance layer is unavailable for a gland-topology guarantee. Hand microscopic gland/lumen quality to generation and post-generation audit.
