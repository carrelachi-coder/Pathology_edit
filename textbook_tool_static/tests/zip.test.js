import assert from "node:assert/strict";
import { buildZip, blobToUint8Array } from "../src/zip.js";

const blob = await buildZip([{ name: "hello.txt", data: "hello" }]);
const bytes = await blobToUint8Array(blob);

assert.equal(bytes[0], 0x50);
assert.equal(bytes[1], 0x4b);
assert.equal(bytes[2], 0x03);
assert.equal(bytes[3], 0x04);
assert.ok(bytes.length > 80);
console.log("ok - buildZip creates a zip-like byte stream");
