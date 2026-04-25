import { acceptanceConflict, nextPendingPatchIndex, queueCounts } from "../src/patchReview.js";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function test(name, fn) {
  try {
    fn();
    console.log(`ok - ${name}`);
  } catch (error) {
    console.error(`not ok - ${name}`);
    console.error(error.message);
    process.exitCode = 1;
  }
}

const patches = [
  { id: "a", x: 0, y: 0, width: 512, height: 512, status: "confirmed" },
  { id: "b", x: 512, y: 0, width: 512, height: 512, status: "pending" },
  { id: "c", x: 256, y: 0, width: 512, height: 512, status: "pending" },
  { id: "d", x: 1024, y: 0, width: 512, height: 512, status: "rejected" }
];

test("acceptanceConflict reports overlap with confirmed patches only", () => {
  const conflict = acceptanceConflict(patches[2], patches);

  assert(conflict?.id === "a", "expected c to conflict with confirmed a");
  assert(acceptanceConflict(patches[1], patches) === null, "expected edge-touching b to be acceptable");
});

test("nextPendingPatchIndex advances through pending queue entries", () => {
  assert(nextPendingPatchIndex(patches, 0) === 1, "expected first pending after a");
  assert(nextPendingPatchIndex(patches, 2) === -1, "expected no pending after c");
  assert(nextPendingPatchIndex(patches, -1) === 1, "expected first pending from start");
});

test("queueCounts summarizes pending, confirmed, and rejected patches", () => {
  const counts = queueCounts(patches);

  assert(counts.pending === 2, "expected two pending patches");
  assert(counts.confirmed === 1, "expected one confirmed patch");
  assert(counts.rejected === 1, "expected one rejected patch");
});
