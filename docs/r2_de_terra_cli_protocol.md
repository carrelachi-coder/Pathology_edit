# Figure 2D/E GPT-5.6 Terra CLI Planner protocol

The `cli-queue` agent mode keeps mask execution on the GPU server while an
account-authenticated Codex CLI on the local Mac performs **only** Planner
decisions. The Semantic Parser remains prebound to the frozen benchmark gold
intent in this D/E experiment. Each Planner call starts a fresh
`codex exec -m gpt-5.6-terra --ephemeral` session; no conversation is resumed.
The CLI sees the existing mask-only candidate packet and schema. It never
receives raw H&E or authority to create interfaces, anchors, pixels, budgets,
cell placements, or new candidates. The normal compiler, executor and hard
gates retain those responsibilities.

Run the local worker on the Mac after the server workflow has created the queue
directory:

```bash
python3 scripts/codex_cli_planner_worker.py \
  --remote-host amax2 \
  --queue-root /data1/lyw/pathology_edit_eval/<version>/terra_queue \
  --codex /Applications/ChatGPT.app/Contents/Resources/codex
```

Use these extra arguments on each server `program_cli` invocation, keeping the
same frozen masks, ProbNet checkpoint and execution flags:

```text
--semantic-parser prebound --agent-mode cli-queue
--planner-queue-root /data1/lyw/pathology_edit_eval/<version>/terra_queue
--model gpt-5.6-terra --reasoning-effort medium
```

The queue stores the full request and response JSON, prompt SHA256, mask-image
SHA256 values, and the fresh CLI session ID. A response with a mismatched
request ID or prompt digest is rejected. The worker removes schema keywords
that Codex's strict-output subset does not accept, then verifies those original
array and numeric constraints on the returned object. The production Planner
also checks every chosen ID against its certified candidate portfolio.

Keep the original 222-request deterministic run intact. Any code-fixed Terra
rerun needs its own versioned output root, code commit, request/cohort hash,
per-request status and independent E raster measurements. Do not omit or
replace abstentions after execution. A 200/222 target cannot be obtained from
the current frozen requests by a Planner change alone: 36 lack required
post-treatment context, six lack a user-specified clearance ROI and six lack
pathology-review authority. Such requests may be redesigned *before* a new
cohort is frozen, but cannot be silently revised inside this comparison.

Pilot proof on 2026-09-23: with the cell-class handoff fix,
`r2de-glas-cell-type-abundance-increase-01` was validated using two independent
Terra CLI sessions. The independent raster audit passed, with 24 requested and
24 added instance events, zero unauthorized tissue transitions and zero changes
outside generation support. This is one integration test, not a cohort result.
