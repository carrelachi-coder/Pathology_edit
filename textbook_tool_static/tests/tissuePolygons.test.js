import { buildMaskFromPolygons, removeLastPolygonForLabel } from "../src/tissuePolygons.js";

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

function at(mask, width, x, y) {
  return mask[y * width + x];
}

test("buildMaskFromPolygons starts as background and fills polygon labels", () => {
  const mask = buildMaskFromPolygons(10, 10, [
    { label: 1, points: [{ x: 1, y: 1 }, { x: 6, y: 1 }, { x: 6, y: 6 }, { x: 1, y: 6 }] }
  ]);

  assert(at(mask, 10, 3, 3) === 1, "expected interior pixel to be tumor");
  assert(at(mask, 10, 8, 8) === 0, "expected outside pixel to remain background");
});

test("later polygons overwrite earlier tissue labels", () => {
  const mask = buildMaskFromPolygons(10, 10, [
    { label: 1, points: [{ x: 1, y: 1 }, { x: 8, y: 1 }, { x: 8, y: 8 }, { x: 1, y: 8 }] },
    { label: 2, points: [{ x: 4, y: 4 }, { x: 9, y: 4 }, { x: 9, y: 9 }, { x: 4, y: 9 }] }
  ]);

  assert(at(mask, 10, 3, 3) === 1, "expected non-overlap to keep first label");
  assert(at(mask, 10, 5, 5) === 2, "expected overlap to use later label");
});

test("removeLastPolygonForLabel removes only the latest polygon for the active label", () => {
  const polygons = [
    { label: 1, points: [{ x: 1, y: 1 }, { x: 8, y: 1 }, { x: 8, y: 8 }, { x: 1, y: 8 }] },
    { label: 2, points: [{ x: 4, y: 4 }, { x: 9, y: 4 }, { x: 9, y: 9 }, { x: 4, y: 9 }] },
    { label: 1, points: [{ x: 0, y: 0 }, { x: 2, y: 0 }, { x: 2, y: 2 }, { x: 0, y: 2 }] }
  ];

  const next = removeLastPolygonForLabel(polygons, 1);
  const mask = buildMaskFromPolygons(10, 10, next);

  assert(next.length === 2, "expected one polygon to be removed");
  assert(next[0].label === 1 && next[1].label === 2, "expected earliest tumor and stroma polygons to remain");
  assert(at(mask, 10, 1, 1) === 1, "expected older tumor polygon to remain");
  assert(at(mask, 10, 5, 5) === 2, "expected later stroma overlap to remain");
});
