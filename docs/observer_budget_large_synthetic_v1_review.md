# Adapter v2 large synthetic B/C trial preparation

This branch prepares a **synthetic-only** target trial. It does not authorize or
start that trial. The previous setup-only authorization was consumed when the
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
are local development evidence only; target environment timing and real
S07 behavior remain untested.
