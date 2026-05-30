import assert from "node:assert/strict";
import { buildZip, blobToUint8Array } from "../src/zip.js";

const blob = await buildZip([
  { name: "hello.txt", data: "hello" },
  { name: "masks/sample_mask.png", data: new Uint8Array([1, 2, 3]) },
  { name: "masks/sample_mask_rgb.png", data: new Uint8Array([4, 5, 6]) }
]);
const bytes = await blobToUint8Array(blob);
const zipText = new TextDecoder("latin1").decode(bytes);

assert.equal(bytes[0], 0x50);
assert.equal(bytes[1], 0x4b);
assert.equal(bytes[2], 0x03);
assert.equal(bytes[3], 0x04);
assert.ok(bytes.length > 80);
assert.ok(zipText.includes("masks/sample_mask.png"));
assert.ok(zipText.includes("masks/sample_mask_rgb.png"));
console.log("ok - buildZip creates a zip-like byte stream");
