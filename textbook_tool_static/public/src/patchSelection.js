export const TISSUE = {
  background: 0,
  tumor: 1,
  stroma: 2,
  necrosis: 3,
  immune: 4,
  normal: 5,
  vessel: 6,
  other: 7
};

const EDITABLE_BOUNDARY_WEIGHTS = new Map([
  [boundaryKey(TISSUE.tumor, TISSUE.stroma), 1.0],
  [boundaryKey(TISSUE.tumor, TISSUE.necrosis), 0.85],
  [boundaryKey(TISSUE.tumor, TISSUE.normal), 0.8],
  [boundaryKey(TISSUE.normal, TISSUE.stroma), 0.65],
  [boundaryKey(TISSUE.tumor, TISSUE.vessel), 0.55]
]);

export function selectBoundaryPatches(mask, width, height, options = {}) {
  const patchSize = options.patchSize ?? 512;
  const stride = options.stride ?? 256;
  const maxPerBucket = options.maxPerBucket ?? 8;
  const candidates = generateCandidates(mask, width, height, patchSize, stride)
    .map((candidate) => scoreCandidate(mask, width, height, candidate))
    .filter((candidate) => candidate.selectionScore > 0 && candidate.backgroundRatio <= 0.3);

  const byBucket = {
    mild: candidates.filter((candidate) => candidate.editScale === "mild"),
    moderate: candidates.filter((candidate) => candidate.editScale === "moderate"),
    large: candidates.filter((candidate) => candidate.editScale === "large")
  };

  for (const bucket of Object.values(byBucket)) {
    bucket.sort((a, b) => b.selectionScore - a.selectionScore);
  }

  const selected = [];
  for (const bucketName of ["mild", "moderate", "large"]) {
    const bucket = byBucket[bucketName].slice(0, maxPerBucket);
    const candidate = bucket.find((item) => selected.every((chosen) => !patchesOverlap(item, chosen)));
    if (candidate) {
      selected.push(candidate);
    }
  }

  const backup = candidates
    .filter((candidate) => selected.every((chosen) => candidate.id !== chosen.id))
    .sort((a, b) => b.selectionScore - a.selectionScore)
    .slice(0, options.backupCount ?? 12);

  return { selected, backup, candidates };
}

export function generateCandidates(mask, width, height, patchSize = 512, stride = 256) {
  const paddedWidth = Math.max(width, patchSize);
  const paddedHeight = Math.max(height, patchSize);
  const xs = slidingStarts(paddedWidth, patchSize, stride);
  const ys = slidingStarts(paddedHeight, patchSize, stride);
  const candidates = [];
  for (const y of ys) {
    for (const x of xs) {
      candidates.push({ id: `${x}_${y}`, x, y, width: patchSize, height: patchSize });
    }
  }
  return candidates;
}

export function scoreCandidate(mask, width, height, candidate) {
  const counts = new Map();
  const boundary = boundaryStats(mask, width, height, candidate);
  const total = candidate.width * candidate.height;
  let validTissue = 0;

  for (let y = candidate.y; y < candidate.y + candidate.height; y += 1) {
    for (let x = candidate.x; x < candidate.x + candidate.width; x += 1) {
      const value = getMask(mask, width, height, x, y);
      counts.set(value, (counts.get(value) ?? 0) + 1);
      if (value !== TISSUE.background) {
        validTissue += 1;
      }
    }
  }

  const tumorRatio = ratio(counts.get(TISSUE.tumor), total);
  const stromaRatio = ratio(counts.get(TISSUE.stroma), total);
  const necrosisRatio = ratio(counts.get(TISSUE.necrosis), total);
  const normalRatio = ratio(counts.get(TISSUE.normal), total);
  const backgroundRatio = ratio(counts.get(TISSUE.background), total);
  const dominantRatio = Math.max(...Array.from(counts.values())) / total;
  const tissueCoverageScore = clamp(validTissue / total / 0.7);
  const editableBoundaryScore = clamp(boundary.weightedLength / (candidate.width * 1.5));
  const boundaryCenterScore = boundary.centerCount === 0 ? 0 : boundary.centerCount / Math.max(1, boundary.editableCount);
  const tissueBalanceScore = clamp(Math.min(tumorRatio, Math.max(stromaRatio, necrosisRatio, normalRatio)) / 0.1);
  const tumorPresenceScore = tumorRatio <= 0 ? 0 : clamp((1 - Math.abs(tumorRatio - 0.35) / 0.35));
  const boundaryIrregularityScore = clamp(boundary.turns / 24);
  const nucleiDensityProxyScore = tissueCoverageScore;

  const backgroundPenalty = clamp((backgroundRatio - 0.15) / 0.25);
  const edgeBoundaryPenalty = boundary.editableCount === 0 ? 0 : boundary.edgeCount / boundary.editableCount;
  const singleTissuePenalty = clamp((dominantRatio - 0.72) / 0.28);
  const maskFragmentationPenalty = clamp(boundary.tinyIslandProxy / 0.12);

  const selectionScore = clamp01(
    0.3 * editableBoundaryScore
      + 0.15 * boundaryCenterScore
      + 0.15 * tissueBalanceScore
      + 0.1 * tumorPresenceScore
      + 0.1 * boundaryIrregularityScore
      + 0.1 * tissueCoverageScore
      + 0.1 * nucleiDensityProxyScore
      - 0.2 * backgroundPenalty
      - 0.15 * edgeBoundaryPenalty
      - 0.15 * singleTissuePenalty
      - 0.1 * maskFragmentationPenalty
  );

  const estimatedChangeRatio = estimateChangeRatio(boundary.editableCount, total);
  return {
    ...candidate,
    boundaryType: boundary.primaryType,
    editableBoundaryLength: boundary.editableCount,
    tumorRatio,
    stromaRatio,
    necrosisRatio,
    normalRatio,
    backgroundRatio,
    estimatedChangeRatio,
    editScale: bucketForChangeRatio(estimatedChangeRatio),
    selectionScore: round(selectionScore)
  };
}

export function patchesOverlap(a, b) {
  const xOverlap = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const yOverlap = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  return xOverlap * yOverlap > 0;
}

export function bucketForChangeRatio(value) {
  if (value >= 0.25) return "large";
  if (value >= 0.15) return "moderate";
  return "mild";
}

function boundaryStats(mask, width, height, candidate) {
  let editableCount = 0;
  let weightedLength = 0;
  let centerCount = 0;
  let edgeCount = 0;
  let turns = 0;
  const typeCounter = new Map();
  const centerMarginX = candidate.width * 0.2;
  const centerMarginY = candidate.height * 0.2;
  const edgeMargin = 64;

  for (let y = candidate.y; y < candidate.y + candidate.height - 1; y += 1) {
    for (let x = candidate.x; x < candidate.x + candidate.width - 1; x += 1) {
      const value = getMask(mask, width, height, x, y);
      const right = getMask(mask, width, height, x + 1, y);
      const down = getMask(mask, width, height, x, y + 1);
      for (const other of [right, down]) {
        const key = boundaryKey(value, other);
        const weight = EDITABLE_BOUNDARY_WEIGHTS.get(key);
        if (!weight) continue;
        editableCount += 1;
        weightedLength += weight;
        typeCounter.set(key, (typeCounter.get(key) ?? 0) + 1);
        const localX = x - candidate.x;
        const localY = y - candidate.y;
        if (
          localX >= centerMarginX
          && localX <= candidate.width - centerMarginX
          && localY >= centerMarginY
          && localY <= candidate.height - centerMarginY
        ) {
          centerCount += 1;
        }
        if (
          localX < edgeMargin
          || localX > candidate.width - edgeMargin
          || localY < edgeMargin
          || localY > candidate.height - edgeMargin
        ) {
          edgeCount += 1;
        }
        if (getMask(mask, width, height, x + 1, y) !== getMask(mask, width, height, x, y + 1)) {
          turns += 1;
        }
      }
    }
  }

  return {
    editableCount,
    weightedLength,
    centerCount,
    edgeCount,
    turns,
    primaryType: primaryBoundaryType(typeCounter),
    tinyIslandProxy: editableCount === 0 ? 0 : turns / Math.max(1, editableCount)
  };
}

function primaryBoundaryType(counter) {
  let best = "none";
  let bestCount = 0;
  for (const [key, count] of counter.entries()) {
    if (count > bestCount) {
      best = key.replace("-", "_");
      bestCount = count;
    }
  }
  return best;
}

function boundaryKey(a, b) {
  const left = Math.min(a, b);
  const right = Math.max(a, b);
  return `${labelName(left)}-${labelName(right)}`;
}

function labelName(value) {
  switch (value) {
    case TISSUE.tumor:
      return "tumor";
    case TISSUE.stroma:
      return "stroma";
    case TISSUE.necrosis:
      return "necrosis";
    case TISSUE.normal:
      return "normal";
    case TISSUE.vessel:
      return "vessel";
    default:
      return String(value);
  }
}

function slidingStarts(length, patchSize, stride) {
  if (length <= patchSize) return [0];
  const starts = [];
  for (let start = 0; start <= length - patchSize; start += stride) {
    starts.push(start);
  }
  const last = length - patchSize;
  if (starts[starts.length - 1] !== last) {
    starts.push(last);
  }
  return starts;
}

function getMask(mask, width, height, x, y) {
  if (x < 0 || y < 0 || x >= width || y >= height) return TISSUE.background;
  return mask[y * width + x] ?? TISSUE.background;
}

function ratio(value, total) {
  return (value ?? 0) / total;
}

function estimateChangeRatio(boundaryLength, total) {
  return clamp01((boundaryLength * 64) / total);
}

function clamp(value) {
  return Math.max(0, Math.min(1, value));
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function round(value) {
  return Math.round(value * 10000) / 10000;
}
