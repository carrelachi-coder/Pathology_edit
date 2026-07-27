import test from "node:test";
import assert from "node:assert/strict";
import {
  buildReviewItems,
  parseMaskFileName,
  parseSelectionManifest,
  pickSelectionManifestFile,
  summarizeReviewItems
} from "../src/maskReview.js";

function file(name, path = "") {
  return { name, webkitRelativePath: path || name, type: "image/png" };
}

test("parseMaskFileName recognizes ID and RGB mask exports", () => {
  assert.deepEqual(parseMaskFileName("case_001_mask.png"), { baseKey: "case_001", kind: "id" });
  assert.deepEqual(parseMaskFileName("case_001_mask_rgb.png"), { baseKey: "case_001", kind: "rgb" });
  assert.equal(parseMaskFileName("case_001.png"), null);
});

test("pickSelectionManifestFile prefers the Chinese selection manifest", () => {
  const files = [
    { name: "notes.csv", type: "text/csv" },
    { name: "selection_manifest_zh.csv", type: "text/csv" },
    { name: "batch_index.csv", type: "text/csv" }
  ];

  assert.equal(pickSelectionManifestFile(files)?.name, "selection_manifest_zh.csv");
  assert.equal(pickSelectionManifestFile([{ name: "notes.txt" }]), null);
});

test("buildReviewItems matches images to ID masks and metadata", () => {
  const images = [file("case_2.png"), file("case 1.png")];
  const masks = [file("case_1_mask_rgb.png"), file("case_1_mask.png")];
  const manifest = parseSelectionManifest("png_file,base_name,organ_zh,caption_zh\ncase_1.png,case_1,肺,描述一\ncase_2.png,case_2,胃,描述二\n");

  const items = buildReviewItems(images, masks, manifest);

  assert.equal(items[0].imageFile.name, "case 1.png");
  assert.equal(items[0].idMask.name, "case_1_mask.png");
  assert.equal(items[0].rgbMask.name, "case_1_mask_rgb.png");
  assert.equal(items[0].metadata, null);
  assert.equal(items[1].status, "missing");
  assert.deepEqual(summarizeReviewItems(items), { total: 2, matched: 1, missing: 1 });
});

test("buildReviewItems preserves manifest matching by original file name", () => {
  const images = [file("case_1.png")];
  const masks = [file("case_1_mask.png")];
  const manifest = parseSelectionManifest("png_file,base_name,organ_zh,caption_zh\ncase_1.png,case_1,肺,描述一\n");

  const items = buildReviewItems(images, masks, manifest);

  assert.equal(items[0].metadata.organZh, "肺");
});
