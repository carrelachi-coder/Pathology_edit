import { equivalentDiameter, medianCellDiameter, remainingCellText } from "../src/cellPolygons.js";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function closeTo(actual, expected, tolerance, message) {
  assert(Math.abs(actual - expected) <= tolerance, `${message}: got ${actual}, expected ${expected}`);
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

test("equivalentDiameter converts polygon area to circular diameter", () => {
  const square = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }];

  closeTo(equivalentDiameter(square), 11.284, 0.01, "expected area-equivalent diameter");
});

test("medianCellDiameter uses completed cell polygons only", () => {
  const cells = [
    { points: [{ x: 0, y: 0 }, { x: 8, y: 0 }, { x: 8, y: 8 }, { x: 0, y: 8 }] },
    { points: [{ x: 0, y: 0 }, { x: 12, y: 0 }, { x: 12, y: 12 }, { x: 0, y: 12 }] }
  ];

  closeTo(medianCellDiameter(cells), 11.284, 0.01, "expected median of equivalent diameters");
});

test("remainingCellText counts down toward ten completed cells", () => {
  assert(remainingCellText(0) === "Cells remaining: 10/10", "expected initial remaining count");
  assert(remainingCellText(1) === "Cells remaining: 9/10", "expected one completed cell to decrement");
  assert(remainingCellText(10) === "Cells remaining: 0/10", "expected target completion count");
});
