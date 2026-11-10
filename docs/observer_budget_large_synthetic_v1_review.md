# Adapter v2 large synthetic B/C trial preparation

This branch began as preparation for a **synthetic-only** target trial. The
previous setup-only authorization was consumed when the
zero-event target preparation passed; the previous S07 authorization was also
consumed. Neither can authorize this workload.

The runner constructs 2,100,000 ordered events per arm, with 683 independently
generated 512 × 512 float64 array occurrences. It records B in hash-only mode
and C in full-capture mode using Adapter v2, validates each stream with the
versioned 30-second read-only worker, then strictly compares B/C with the
versioned 60-second worker. It independently regenerates every C array and
checks its original bytes. It never imports or calls a pathology planner,
guard, witness, S07 source, or H&E generator. It never publishes the outer
experimental acceptance marker.

Before any target run, freeze a fresh manifest with exact code hashes,
interpreter and distribution identity, environment, output paths, workload,
storage budget, setup and arm timeouts, and an overall OS watchdog. Obtain a
separate approval bound to that manifest SHA. Each stage runs once. Any
identity mismatch, timeout, invalid evidence, or strict comparison failure
is a permanent HOLD for that authorization.

This trial measures only the fixed synthetic workload in the target
environment. Passing it would not establish real S07 event count, raster
capacity, non-interference, topology qualification, or formal cohort
eligibility.

## Local full-workload development results

On the local Mac, one run using the full 2,100,000-event workload per arm
completed with a strict semantic B/C match. B and C each recorded 683
generated arrays plus two RNG checkpoint occurrences. Independently
regenerated C payloads matched all 683 generated arrays byte for byte.
The full run took 84.89 seconds; B/C comparison took 16.66 seconds. B
allocated 15.72 MB and C 1.45 GB, both below 4 GiB. The compact,
hash-bound result is in
`benchmarks/manuscript_validation_20261109/adapter_controller_v1/observer_budget_large_synthetic_v1/local_full_workload_result_v1.json`.
The setup boundary was then corrected to include Adapter and validator
imports. A separate full-workload run under the corrected code again
produced a strict semantic match and independently matched all 683 C
arrays. It took 109.60 seconds overall; setup was 0.21 seconds and B/C
comparison was 24.19 seconds. The second result is
`benchmarks/manuscript_validation_20261109/adapter_controller_v1/observer_budget_large_synthetic_v1/local_full_workload_result_v2.json`.
The first result remains archived with its original code identity. Both
are local development evidence only; neither established target timing
or real S07 behavior.

## One authorized target trial

The user separately approved the frozen launch manifest SHA256
`91ffabf1ae6a45e7270328fde2ad746649e11884a8ad658b187768ccb44ff5c5`.
The one-shot target run ended in **HOLD** after 193.67 seconds: B hit the
180-second generation deadline before completing its event loop. No B
capture was finalized, C and B/C comparison did not start, and no outer
acceptance proof was published. The partial B index has 25 block rows
through sequence 978,493; the 25 referenced block files match their
SHA256 values. The index is still pending, so this is partial physical
evidence, not a complete business record. S07, witness and H&E calls
were zero. The authorization is consumed and will not be reused.

The immutable terminal and partial-evidence summary is
`benchmarks/manuscript_validation_20261109/adapter_controller_v1/observer_budget_large_synthetic_v1/target_hold_result_v1/closeout_report.json`.
This HOLD does not show a Codec or comparison mismatch; neither full
capture nor comparison ran. The target workload rate and future
engineering budget need separate analysis before proposing another
trial.

## Bounded budget v2 target closeout

The user reviewed the synthetic-only engineering budget revision and
separately authorized one run bound to manifest SHA256
`cd90f2839e5aaec74beb459e38a9cb3d412af0e2ccf65259815b35d9ee76173d`.
The old HOLD remains unchanged. The v2 run also ended in **HOLD**, after
798.08 seconds. B completed all 2,100,000 events and passed its
30-second single-stream validator in 19.80 seconds. C then reached the
480-second generation deadline before finishing. Its pending block
index has 36 rows through sequence 1,404,893; all 36 referenced block
files matched their SHA256 values. C has no codec completion marker, so
this partial physical evidence is not a valid complete capture. Strict
B/C comparison never started, and no outer acceptance proof was
published. The authorization is consumed; S07, witness, H&E and
automatic retry counts were zero.

The terminal evidence is
`benchmarks/manuscript_validation_20261109/adapter_controller_v1/observer_budget_large_synthetic_v1/target_hold_result_v2/closeout_report.json`.
This establishes a target-environment Full Capture throughput
constraint for the frozen synthetic workload. It does not establish a
B/C mismatch or a real S07 runtime bound.
