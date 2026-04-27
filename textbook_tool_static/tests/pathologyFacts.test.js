import assert from "node:assert/strict";
import { buildPatchPathologyFacts, cropMaskToPatch } from "../src/pathologyFacts.js";

function test(name, fn) {
  try {
    fn();
    console.log(`ok - ${name}`);
  } catch (error) {
    console.error(`not ok - ${name}`);
    throw error;
  }
}

function fillRect(mask, width, x0, y0, w, h, value) {
  for (let y = y0; y < y0 + h; y += 1) {
    for (let x = x0; x < x0 + w; x += 1) {
      mask[y * width + x] = value;
    }
  }
}

test("cropMaskToPatch pads out-of-bounds areas as background", () => {
  const source = new Uint8Array(4 * 4).fill(2);
  const patch = cropMaskToPatch(source, 4, 4, { x: 2, y: 2, width: 4, height: 4 });

  assert.equal(patch.width, 4);
  assert.equal(patch.height, 4);
  assert.equal(patch.mask[0], 2);
  assert.equal(patch.mask[1], 2);
  assert.equal(patch.mask[4], 2);
  assert.equal(patch.mask[5], 2);
  assert.equal(patch.mask[2], 0);
  assert.equal(patch.mask[15], 0);
});

test("buildPatchPathologyFacts summarizes tissue composition boundaries and nuclei", () => {
  const width = 8;
  const height = 8;
  const tissueMask = new Uint8Array(width * height);
  const nucleiMask = new Uint8Array(width * height);
  fillRect(tissueMask, width, 0, 0, 4, 8, 1);
  fillRect(tissueMask, width, 4, 0, 3, 8, 2);
  fillRect(tissueMask, width, 7, 0, 1, 8, 4);
  nucleiMask[1 * width + 1] = 101;
  nucleiMask[2 * width + 2] = 101;
  nucleiMask[1 * width + 5] = 102;

  const facts = buildPatchPathologyFacts({
    sampleId: "book01_py0_px0",
    organ: "colorectal",
    cancerType: "colorectal adenocarcinoma",
    globalDescription: "Tumor adjacent to stroma.",
    tissueMask,
    nucleiMask,
    width,
    height,
    boundaryType: "tumor_stroma",
    editScale: "moderate",
    recommendedEditType: "tumor_expansion_into_stroma",
    changeRatioTarget: 0.18
  });

  assert.equal(facts.sample_id, "book01_py0_px0");
  assert.deepEqual(facts.dominant_tissues, ["tumor", "stroma", "immune"]);
  assert.equal(facts.tissue_composition.tumor.pixel_count, 32);
  assert.equal(facts.tissue_composition.stroma.pixel_count, 24);
  assert.equal(facts.tissue_composition.immune.pixel_count, 8);
  assert.equal(facts.tissue_boundary.primary_boundary_type, "tumor_stroma");
  assert.equal(facts.tissue_boundary.boundary_pair_counts.tumor_stroma, 8);
  assert.equal(facts.tissue_boundary.boundary_pair_counts.stroma_immune, 8);
  assert.equal(facts.nuclei_counts.neoplastic, 2);
  assert.equal(facts.nuclei_counts.inflammatory, 1);
  assert.equal(facts.nuclei_density_level.neoplastic, "high");
  assert.equal(facts.pathology_context.organ, "colorectal");
  assert.equal(facts.pathology_context.cancer_type, "colorectal adenocarcinoma");
});
