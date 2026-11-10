# Synthetic Codec array path microprofile

The first target large synthetic B/C trial stopped in B at 180 seconds.
The bounded second trial completed and validated B, then stopped during C
Full Capture at 480 seconds. Both one-shot authorizations are consumed.
Neither trial reached strict B/C comparison, and neither called S07,
pathology witnesses or H&E generation.

The direct Codec microprofile in this branch is a new, separate proposal.
It runs 100,000 deterministic events and 32 independently generated
512 × 512 float64 arrays in each mode. B uses Hash-only and C uses Full
Capture. Each mode has its own cProfile output; the script then checks
both complete Codec streams for a strict match. The original Codec and
Adapter code are unchanged. The script does not import pathology modules.

The target proposal fixes a 60-second setup, 120 seconds for each
synthetic encode/seal arm, and a 360-second outer OS watchdog. Failure
is HOLD with no automatic retry. It uses new isolated paths and needs
separate code review, a final versioned launch manifest and a new
one-shot authorization before any target run.

Local tests: three focused cases passed. A local full candidate workload
completed with strict B/C equality, unchanged global RNG and zero real
source calls. The report is
`benchmarks/manuscript_validation_20261109/adapter_controller_v1/observer_array_microprofile_v1/local_validation_report_v2.json`.
Local timings do not establish target filesystem behavior.
