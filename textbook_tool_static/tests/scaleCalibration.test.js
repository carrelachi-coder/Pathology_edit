import { calibrationReferenceDiameter, calibrationTypeLabel, availableCalibrationTypes } from "../src/scaleCalibration.js";

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

const summary = {
  neoplastic_diameter_px_median: 31,
  nucleus_diameter_px_median: 24,
  type_stats: {
    "101": { stored_count: 100, nucleus_diameter_px_median: 31 },
    "102": { stored_count: 80, nucleus_diameter_px_median: 18 },
    "103": { stored_count: 60, nucleus_diameter_px_median: 20 }
  }
};

test("calibrationReferenceDiameter prefers the selected nuclei type", () => {
  assert(calibrationReferenceDiameter(summary, "102") === 18, "expected inflammatory reference");
  assert(calibrationReferenceDiameter(summary, "103") === 20, "expected connective reference");
});

test("calibrationReferenceDiameter falls back to neoplastic then global median", () => {
  assert(calibrationReferenceDiameter(summary, "105") === 31, "expected neoplastic fallback");
  assert(calibrationReferenceDiameter({ nucleus_diameter_px_median: 22 }, "105") === 22, "expected global fallback");
});

test("availableCalibrationTypes returns only types present in summary", () => {
  const types = availableCalibrationTypes(summary).map((item) => item.id);

  assert(types.join(",") === "101,102,103", "expected available type ids only");
});

test("calibrationTypeLabel marks neoplastic as preferred", () => {
  assert(calibrationTypeLabel("101").includes("preferred"), "expected preferred tumor label");
  assert(calibrationTypeLabel("102").includes("inflammatory"), "expected inflammatory label");
});
