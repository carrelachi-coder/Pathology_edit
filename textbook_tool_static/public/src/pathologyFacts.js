export const TISSUE_FACT_LABELS = new Map([
  [0, "background"],
  [1, "tumor"],
  [2, "stroma"],
  [3, "necrosis"],
  [4, "immune"],
  [5, "normal"],
  [6, "vessel"],
  [7, "other"]
]);

export const NUCLEI_FACT_LABELS = new Map([
  [101, "neoplastic"],
  [102, "inflammatory"],
  [103, "connective"],
  [104, "dead"],
  [105, "epithelial"]
]);

const FACTS_VERSION = "0.1.0";
const NORMALIZED_PATCH_AREA = 512 * 512;

export function cropMaskToPatch(mask, width, height, patch, background = 0) {
  const output = new Uint8Array(patch.width * patch.height);
  for (let y = 0; y < patch.height; y += 1) {
    for (let x = 0; x < patch.width; x += 1) {
      const sourceX = patch.x + x;
      const sourceY = patch.y + y;
      output[y * patch.width + x] = (
        sourceX >= 0 && sourceY >= 0 && sourceX < width && sourceY < height
      ) ? mask[sourceY * width + sourceX] : background;
    }
  }
  return { mask: output, width: patch.width, height: patch.height };
}

export function buildPatchPathologyFacts({
  sampleId,
  organ,
  cancerType,
  globalDescription,
  tissueMask,
  nucleiMask,
  width,
  height,
  boundaryType,
  editScale,
  recommendedEditType,
  changeRatioTarget
}) {
  const totalPixels = width * height;
  const tissueComposition = summarizeLabels(tissueMask, totalPixels, TISSUE_FACT_LABELS);
  const nucleiCounts = countLabels(nucleiMask, NUCLEI_FACT_LABELS);
  const boundaryPairCounts = countBoundaryPairs(tissueMask, width, height);
  const nucleiDensity = {};
  const nucleiDensityLevel = {};
  for (const name of NUCLEI_FACT_LABELS.values()) {
    const count = nucleiCounts[name] ?? 0;
    const perPatch = count * (NORMALIZED_PATCH_AREA / Math.max(1, totalPixels));
    nucleiDensity[name] = round(perPatch);
    nucleiDensityLevel[name] = densityLevel(perPatch);
  }

  return {
    pathology_facts_version: FACTS_VERSION,
    facts_source: "mask_derived",
    sample_id: sampleId,
    pathology_context: {
      organ: organ || "",
      cancer_type: cancerType || "",
      global_description: globalDescription || ""
    },
    edit_context: {
      edit_scale: editScale || "",
      recommended_edit_type: recommendedEditType || "",
      change_ratio_target: changeRatioTarget ?? null
    },
    dominant_tissues: dominantTissues(tissueComposition),
    tissue_composition: tissueComposition,
    tissue_boundary: {
      primary_boundary_type: boundaryType || primaryBoundaryType(boundaryPairCounts),
      boundary_pair_counts: boundaryPairCounts
    },
    nuclei_counts: nucleiCounts,
    nuclei_density_per_512_patch: nucleiDensity,
    nuclei_density_level: nucleiDensityLevel
  };
}

function summarizeLabels(mask, totalPixels, labels) {
  const counts = countLabels(mask, labels);
  const composition = {};
  for (const name of labels.values()) {
    const pixelCount = counts[name] ?? 0;
    composition[name] = {
      pixel_count: pixelCount,
      ratio: round(pixelCount / Math.max(1, totalPixels))
    };
  }
  return composition;
}

function countLabels(mask, labels) {
  const counts = {};
  for (const name of labels.values()) {
    counts[name] = 0;
  }
  for (const value of mask) {
    const name = labels.get(value);
    if (!name) continue;
    counts[name] += 1;
  }
  return counts;
}

function dominantTissues(composition) {
  return Object.entries(composition)
    .filter(([name, stats]) => name !== "background" && stats.ratio >= 0.01)
    .sort((a, b) => b[1].ratio - a[1].ratio)
    .map(([name]) => name);
}

function countBoundaryPairs(mask, width, height) {
  const counts = {};
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = mask[y * width + x];
      if (x + 1 < width) {
        addBoundaryPair(counts, value, mask[y * width + x + 1]);
      }
      if (y + 1 < height) {
        addBoundaryPair(counts, value, mask[(y + 1) * width + x]);
      }
    }
  }
  return counts;
}

function addBoundaryPair(counts, a, b) {
  if (a === b) return;
  const left = labelName(Math.min(a, b));
  const right = labelName(Math.max(a, b));
  if (left === "background" && right === "background") return;
  const key = `${left}_${right}`;
  counts[key] = (counts[key] ?? 0) + 1;
}

function primaryBoundaryType(counts) {
  let best = "none";
  let bestCount = 0;
  for (const [key, count] of Object.entries(counts)) {
    if (count > bestCount) {
      best = key;
      bestCount = count;
    }
  }
  return best;
}

function labelName(value) {
  return TISSUE_FACT_LABELS.get(value) || `label_${value}`;
}

function densityLevel(value) {
  if (value <= 0) return "none";
  if (value >= 80) return "high";
  if (value >= 25) return "moderate";
  return "low";
}

function round(value) {
  return Math.round(value * 10000) / 10000;
}
