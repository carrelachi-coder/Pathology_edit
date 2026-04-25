export const CALIBRATION_TYPES = [
  { id: "101", name: "tumor / neoplastic", preferred: true },
  { id: "102", name: "inflammatory", preferred: false },
  { id: "103", name: "connective", preferred: false },
  { id: "104", name: "dead", preferred: false },
  { id: "105", name: "epithelial", preferred: false }
];

export function calibrationTypeLabel(id) {
  const type = CALIBRATION_TYPES.find((item) => item.id === String(id));
  if (!type) return String(id);
  return type.preferred ? `${type.name} (preferred)` : type.name;
}

export function availableCalibrationTypes(summary) {
  return CALIBRATION_TYPES.filter((type) => (
    summary?.type_stats?.[type.id]?.nucleus_diameter_px_median
    && (summary.type_stats[type.id].stored_count ?? 1) > 0
  ));
}

export function calibrationReferenceDiameter(summary, selectedTypeId = "101") {
  const selected = summary?.type_stats?.[String(selectedTypeId)]?.nucleus_diameter_px_median;
  if (Number.isFinite(selected)) return selected;

  const neoplastic = summary?.type_stats?.["101"]?.nucleus_diameter_px_median
    ?? summary?.neoplastic_diameter_px_median;
  if (Number.isFinite(neoplastic)) return neoplastic;

  const global = summary?.nucleus_diameter_px_median;
  return Number.isFinite(global) ? global : 25;
}
