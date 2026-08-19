# Breast Golden + GLaS/PANDA Consolidation Plan — V2

## What changed

The newly supplied `panel_manifest(5).json` is a unique 25-case five-edit
review bundle generated at:

`20e8600910adb935f58333c3a4235179199331fe`

It covers five cases each for:

- Generic immune increase
- Generic immune decrease
- Cell-type abundance increase
- Cell-type abundance decrease
- Cellularity increase

Twenty-four cases are `validated_first_pass`; one is `recovered`. All 25
masks passed their hard gates and all 25 images are source-exact outside the
approved support.

Therefore `codex/fix-cellularity-ledger-20260813` is a mandatory golden
donor, not a merely speculative branch.

## De-duplicated artifact inventory

There are five unique manifest groups and 85 unique panels:

1. Five cell/immune edits — 25
2. Four treatment/necrosis edits — 20
3. Scatter/Cluster/Neoplastic increase — 15
4. Cord/Detached Nest — 10
5. Three post-treatment edits — 15

The other four newly uploaded files are exact duplicates of the earlier
four manifests.

## Integration branch

Create:

`integration/breast-golden-glas-panda-20260819`

No existing branch is the complete golden branch.

Use `071c3422...` only as a clean-history root, then squash-import a cleaned
checkpoint of the current GLaS/PANDA tree. The semantic golden behavior
comes from all mandatory donor lines, especially `20e860...`.

## Mandatory donor order

### 1. Five-edit line first

Source:

`codex/fix-cellularity-ledger-20260813`
`20e8600910adb935f58333c3a4235179199331fe`

Preserve:

- Generic immune increase: Inpaint
- Generic immune decrease: Cross
- Cell-type abundance increase: mostly Inpaint, one recovered Cross
- Cell-type abundance decrease: Cross
- Cellularity increase: Inpaint
- Exact source preservation outside support
- Mask hard-gate binding
- Cell-only evaluator/executor ledger behavior

The current generic-immune joint-gate failures must be treated as
regressions against this 25-case baseline. Do not solve them by lowering
counts to the currently executable minimum.

### 2. Cord and detached Nest

Source:

`codex/retire-tumor-burden-primitive`

Golden commits:

- Cord: `c4e02f18...`
- Detached Nest: `95c4bfcd...`

Preserve their visible architecture scale and wider generation context.

### 3. Four treatment/necrosis edits

Source:

`codex/four-edit-distinct-20260814`
`c2510b650...`

Port capability-specific behavior and frozen cases; do not overwrite the
shared runtime wholesale.

### 4. Current-line artifact groups

Scatter/Cluster/Neoplastic increase and the three post-treatment edits are
already on the ancestry of the current GLaS/PANDA line. Keep their frozen
masks, image hashes, J/G routing records, and original review statuses.

## Clean checkpoint procedure

```bash
cd /Users/wangqinxin/Documents/GitHub/Pathology_edit
git fetch origin
git switch -c checkpoint/glas-panda-current-20260819

rm -rf .glas_*_backup_* .joint_*_backup_*
rm -f glas_*_applied.json joint_*_applied.json
```

Add:

```gitignore
.glas_*_backup_*/
.joint_*_backup_*/
glas_*_applied.json
joint_*_applied.json
```

Commit only intentional source, skill, resource, script, and test files.

## Clean integration worktree

```bash
git worktree add       -b integration/breast-golden-glas-panda-20260819       ../Pathology_edit_golden       071c3422dcbb4c3ce86b5e6a787b3415587df00f

cd ../Pathology_edit_golden
git merge --squash checkpoint/glas-panda-current-20260819
git commit -m "Import curated current Breast, GLaS and PANDA baseline"
```

Add read-only donor worktrees:

```bash
git worktree add ../Pathology_edit_five_edits       codex/fix-cellularity-ledger-20260813

git worktree add ../Pathology_edit_cord_nest       codex/retire-tumor-burden-primitive

git worktree add ../Pathology_edit_four_edits       codex/four-edit-distinct-20260814
```

## Promotion gates

- Index all 85 unique panels and their hashes.
- Preserve all 25 five-edit masks and outside-support exactness.
- Restore the 20e860 direction-specific generation routing.
- Reproduce the 10 Cord/Nest validated panels without reducing visible effect.
- Reproduce the four treatment/necrosis fixed masks.
- Preserve GLaS authority and gland/nucleus digest separation.
- Preserve PANDA pattern/fine-label and auxiliary-digest authority.
- Keep uncertain/review statuses distinct from automatic validation.
- Pass Breast fixed-case mask and H&E regressions before full joint tests.
- Only after all gates pass, promote to:
  `release/breast-bcss-glas-panda-v1`.
