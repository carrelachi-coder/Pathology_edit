import { patchesOverlap } from "./patchSelection.js";

export function acceptanceConflict(patch, patches) {
  return patches.find((item) => (
    item.id !== patch.id
    && item.status === "confirmed"
    && patchesOverlap(item, patch)
  )) || null;
}

export function nextPendingPatchIndex(patches, currentIndex = -1) {
  for (let index = currentIndex + 1; index < patches.length; index += 1) {
    if (patches[index].status === "pending") return index;
  }
  return -1;
}

export function queueCounts(patches) {
  return patches.reduce((counts, patch) => {
    counts[patch.status] = (counts[patch.status] ?? 0) + 1;
    return counts;
  }, { pending: 0, accepted: 0, confirmed: 0, rejected: 0 });
}
