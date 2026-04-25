import assert from "node:assert/strict";
import { TISSUE, bucketForChangeRatio, patchesOverlap, selectBoundaryPatches } from "../src/patchSelection.js";

function test(name, fn) {
  try {
    fn();
    console.log(`ok - ${name}`);
  } catch (error) {
    console.error(`not ok - ${name}`);
    throw error;
  }
}

test("bucketForChangeRatio assigns mild moderate and large buckets", () => {
  assert.equal(bucketForChangeRatio(0.05), "mild");
  assert.equal(bucketForChangeRatio(0.15), "moderate");
  assert.equal(bucketForChangeRatio(0.25), "large");
});

test("patchesOverlap detects any pixel intersection", () => {
  const a = { x: 0, y: 0, width: 10, height: 10 };
  assert.equal(patchesOverlap(a, { x: 10, y: 0, width: 10, height: 10 }), false);
  assert.equal(patchesOverlap(a, { x: 9, y: 0, width: 10, height: 10 }), true);
});

test("selectBoundaryPatches enforces zero overlap for selected patches", () => {
  const width = 768;
  const height = 512;
  const mask = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      mask[y * width + x] = x < width / 2 ? TISSUE.tumor : TISSUE.stroma;
    }
  }

  const result = selectBoundaryPatches(mask, width, height, {
    patchSize: 256,
    stride: 128,
    maxPerBucket: 20
  });

  for (let i = 0; i < result.selected.length; i += 1) {
    for (let j = i + 1; j < result.selected.length; j += 1) {
      assert.equal(patchesOverlap(result.selected[i], result.selected[j]), false);
    }
  }
});
