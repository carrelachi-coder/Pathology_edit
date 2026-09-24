# Figure 2D/E source-qualified follow-up cohort (v4)

Version 4 is a **new, frozen cohort**, developed after auditing the completed
v3 results. It must be reported separately from v3 (141/222 after same-case
retries). It is not a corrected estimate for the v3 denominator. The goal is
to test mask-aware planning and execution on requests whose **source masks**
meet predefined necessary conditions. Every v3 patch is excluded. Neither a
Planner answer, executed edit, validation gate, nor independent E result is
used to select a v4 patch. The fixed builder is
`scripts/r2_de_six_v4_freeze_cohort.py`, and its complete cohort, selection
audit, original protocol, and pre-execution timeout amendment are under
`benchmarks/r2_de_six_v4/`.

The cohort contains 222 requests, 37 per dataset. BCSS has 27 tumor-boundary
expansions and 10 immune-compartment decreases. GlaS and ORCA each have 37
local cellularity decreases. PANDA has 10 inflammatory-cell decreases and 27
tumor-boundary expansions. IGNITE has 27 inflammatory-cell decreases and 10
tumor-boundary expansions. PUMA has 8 inflammatory-cell decreases, 9
tumor-boundary expansions, 15 inflammatory-cell increases, and 5
neoplastic-cell decreases. The PUMA mix is broader because the available
primary-melanoma patches left after excluding v3 cannot support 37 distinct
requests of the two original types under the new source screen. Twelve PUMA
patches each receive two **different** edit requests; no patch has more than
two. The cohort has 210 unique dataset–patch pairs and 222 requests. All
results require patch- and source-group-aware interpretation.

The source-only screen now checks complete target-class instances within a
single tissue component and their spatial span, in addition to coarse nuclei
or tissue area. It checks accessible stroma near the intended external tumor
front and a larger immune compartment for BCSS. PANDA tumor growth requires a
source Pattern-4 or Pattern-5 external front: fine ID 9 or 10, respectively.
Its natural-language request names the existing pattern and asks to preserve
other fine patterns. This prevents Pattern-3-only patches (fine ID 8) from
being presented as Pattern-4/5 growth requests. These are necessary source
conditions, **not certificates of successful editing**; the ordinary
compiler, Planner, executor, and all hard gates still run on every frozen
request. The 27 PANDA growth requests retain model choice among eligible
interfaces and anchors within the specified pattern.

The area budget for v4 tissue requests is 4% target, 2.6% minimum, 6% maximum,
and 2.6% tissue minimum. This is a prospectively fixed, still visible edit
range. Cell-only requests retain the same skill-owned minimum effects and
profile-derived budgets as the code; the builder does not lower cell minima.
The frozen manifest does not contain doctor labels or use H&E pixels for
request selection beyond requiring the source image to exist and align with
the masks. The prebound semantic request isolates Planner and executor
performance rather than Parser accuracy. The program uses fresh GPT-5.6
Terra CLI sessions, the frozen mature ProbNet checkpoint, and independent E
raster checks for every validated output.

The cohort was frozen before the first v4 execution. The original protocol
specified a 600-second wall-clock cap; before any case ran, the cap was
amended to 1800 seconds to accommodate source preparation and the Terra
queue. `protocol_frozen_initial.json` preserves the original setting.
No cohort member may be replaced or altered after observing a result. Any
software repair requires an explicit versioned amendment and a numbered
same-case retry linked to the original outcome. Report initial and recovered
rates separately, per dataset and per primitive, including failures and
abstentions. Independent E addresses mask-contract preservation only; it
cannot establish H&E realism or diagnostic validity.

The source-group IDs are filename-derived, not verified patients. PANDA's
Radboud-style label schema does not prove slide-acquisition institution, and
PUMA's exact skin subsite is unavailable. Both limitations carry over from
the v3 protocol.
