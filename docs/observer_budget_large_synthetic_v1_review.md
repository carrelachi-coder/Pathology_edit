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
