import { TISSUE, selectBoundaryPatches, patchesOverlap } from "./patchSelection.js";
import { buildZip } from "./zip.js";

const tissueLabels = [
  ["background", 0, "#000000"],
  ["tumor", 1, "#d92d20"],
  ["stroma", 2, "#039855"],
  ["necrosis", 3, "#4e5ba6"],
  ["immune", 4, "#7a5af8"],
  ["normal", 5, "#fdb022"],
  ["vessel", 6, "#dc6803"],
  ["other", 7, "#667085"]
];

const nucleiLabels = [
  ["neoplastic", 101, "#d92d20"],
  ["inflammatory", 102, "#12b76a"],
  ["connective", 103, "#1570ef"],
  ["dead", 104, "#fdb022"],
  ["epithelial", 105, "#c11574"]
];

const state = {
  image: null,
  imageBitmap: null,
  imageName: "",
  metadata: {},
  tissueMask: null,
  tumorCells: [],
  normalized: null,
  patches: [],
  acceptedPatchIds: new Set(),
  selectedPatchId: null,
  nucleiByPatch: new Map(),
  mode: "tissue",
  tissueLabel: 1,
  nucleiLabel: 101,
  librarySummary: null
};

const els = {};
for (const id of [
  "downloadZip", "imageInput", "metadataInput", "imageId", "organ", "cancerType",
  "libraryKey", "globalDescription", "tissueLabels", "brushSize", "clearTissue",
  "cellRadius", "cellMode", "clearCells", "scaleStatus", "normalizeAndSelect",
  "nucleiLabels", "nucleusRadius", "tissueMode", "reviewMode", "nucleiMode",
  "status", "mainCanvas", "patchGrid"
]) {
  els[id] = document.getElementById(id);
}

const ctx = els.mainCanvas.getContext("2d", { willReadFrequently: true });
let isDrawing = false;

init();

function init() {
  renderLabelButtons(els.tissueLabels, tissueLabels, "tissue");
  renderLabelButtons(els.nucleiLabels, nucleiLabels, "nuclei");
  bindEvents();
  loadLibrarySummary();
  setStatus("Load an image to begin.");
}

function bindEvents() {
  els.imageInput.addEventListener("change", handleImageInput);
  els.metadataInput.addEventListener("change", handleMetadataInput);
  els.libraryKey.addEventListener("change", loadLibrarySummary);
  els.clearTissue.addEventListener("click", () => {
    if (!state.tissueMask) return;
    state.tissueMask.fill(0);
    drawMainCanvas();
  });
  els.cellMode.addEventListener("click", () => setMode("cell"));
  els.clearCells.addEventListener("click", () => {
    state.tumorCells = [];
    updateScaleStatus();
    drawMainCanvas();
  });
  els.normalizeAndSelect.addEventListener("click", normalizeAndSelect);
  els.tissueMode.addEventListener("click", () => setMode("tissue"));
  els.reviewMode.addEventListener("click", () => setMode("review"));
  els.nucleiMode.addEventListener("click", () => setMode("nuclei"));
  els.downloadZip.addEventListener("click", downloadZip);

  els.mainCanvas.addEventListener("pointerdown", handlePointerDown);
  els.mainCanvas.addEventListener("pointermove", handlePointerMove);
  window.addEventListener("pointerup", () => {
    isDrawing = false;
  });
}

function renderLabelButtons(container, labels, kind) {
  container.replaceChildren();
  for (const [name, value, color] of labels) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${name} (${value})`;
    button.style.borderLeft = `8px solid ${color}`;
    if ((kind === "tissue" && value === state.tissueLabel) || (kind === "nuclei" && value === state.nucleiLabel)) {
      button.classList.add("active");
    }
    button.addEventListener("click", () => {
      if (kind === "tissue") state.tissueLabel = value;
      if (kind === "nuclei") state.nucleiLabel = value;
      renderLabelButtons(container, labels, kind);
    });
    container.appendChild(button);
  }
}

async function handleImageInput(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  state.imageName = file.name;
  state.image = await createImageBitmap(file);
  state.imageBitmap = state.image;
  state.tissueMask = new Uint8Array(state.image.width * state.image.height);
  state.tumorCells = [];
  state.normalized = null;
  state.patches = [];
  state.acceptedPatchIds.clear();
  state.nucleiByPatch.clear();
  els.imageId.value = file.name.replace(/\.[^.]+$/, "");
  resizeCanvas(state.image.width, state.image.height);
  drawMainCanvas();
  renderPatches();
  updateScaleStatus();
  setStatus(`Loaded ${file.name} (${state.image.width}x${state.image.height}).`);
}

async function handleMetadataInput(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const payload = JSON.parse(await file.text());
  const record = Array.isArray(payload) ? payload[0] : payload;
  state.metadata = record;
  els.imageId.value = record.image_id || els.imageId.value;
  els.organ.value = record.organ || els.organ.value;
  els.cancerType.value = record.cancer_type || els.cancerType.value;
  els.libraryKey.value = record.nuclei_library_key || record.organ || els.libraryKey.value;
  els.globalDescription.value = record.global_description || "";
  await loadLibrarySummary();
  setStatus(`Loaded metadata for ${els.imageId.value}.`);
}

async function loadLibrarySummary() {
  const key = els.libraryKey.value;
  try {
    const response = await fetch(`./nuclei_library_stats/${key}.json`);
    if (!response.ok) throw new Error(`missing ${key}.json`);
    state.librarySummary = await response.json();
    updateScaleStatus();
  } catch {
    state.librarySummary = null;
    els.scaleStatus.textContent = `No nuclei summary found for ${key}.`;
  }
}

function setMode(mode) {
  state.mode = mode;
  for (const [button, name] of [[els.tissueMode, "tissue"], [els.reviewMode, "review"], [els.nucleiMode, "nuclei"]]) {
    button.classList.toggle("active", mode === name);
  }
  els.cellMode.classList.toggle("active", mode === "cell");
  drawMainCanvas();
}

function handlePointerDown(event) {
  if (!state.imageBitmap) return;
  const point = canvasPoint(event);
  if (state.mode === "cell") {
    addTumorCell(point);
    return;
  }
  if (state.mode === "nuclei" && state.selectedPatchId) {
    addPatchNucleus(point);
    return;
  }
  if (state.mode === "tissue") {
    isDrawing = true;
    paintTissue(point);
  }
}

function handlePointerMove(event) {
  if (!isDrawing || state.mode !== "tissue") return;
  paintTissue(canvasPoint(event));
}

function canvasPoint(event) {
  const rect = els.mainCanvas.getBoundingClientRect();
  return {
    x: Math.floor((event.clientX - rect.left) * (els.mainCanvas.width / rect.width)),
    y: Math.floor((event.clientY - rect.top) * (els.mainCanvas.height / rect.height))
  };
}

function paintTissue(point) {
  const radius = Number(els.brushSize.value);
  paintCircle(state.tissueMask, state.image.width, state.image.height, point.x, point.y, radius, state.tissueLabel);
  drawMainCanvas();
}

function addTumorCell(point) {
  state.tumorCells.push({ x: point.x, y: point.y, radius: Number(els.cellRadius.value) });
  updateScaleStatus();
  drawMainCanvas();
}

function addPatchNucleus(point) {
  const patch = state.patches.find((item) => item.id === state.selectedPatchId);
  if (!patch) return;
  const local = { x: point.x - patch.x, y: point.y - patch.y };
  if (local.x < 0 || local.y < 0 || local.x >= patch.width || local.y >= patch.height) return;
  const nuclei = state.nucleiByPatch.get(patch.id) || [];
  nuclei.push({ x: local.x, y: local.y, radius: Number(els.nucleusRadius.value), label: state.nucleiLabel });
  state.nucleiByPatch.set(patch.id, nuclei);
  drawMainCanvas();
}

async function normalizeAndSelect() {
  if (!state.imageBitmap || !state.tissueMask) return;
  const scale = estimateScaleFactor();
  const width = Math.max(1, Math.round(state.image.width * scale));
  const height = Math.max(1, Math.round(state.image.height * scale));
  const imageCanvas = drawScaledImage(state.imageBitmap, width, height);
  const mask = scaleMaskNearest(state.tissueMask, state.image.width, state.image.height, width, height);
  const selection = selectBoundaryPatches(mask, width, height, { patchSize: 512, stride: 256 });
  state.normalized = { imageCanvas, mask, width, height, scaleFactor: scale };
  state.patches = [...selection.selected, ...selection.backup].map((patch, index) => ({
    ...patch,
    accepted: index < selection.selected.length,
    source: index < selection.selected.length ? "selected" : "backup"
  }));
  state.acceptedPatchIds = new Set(state.patches.filter((patch) => patch.accepted).map((patch) => patch.id));
  resizeCanvas(width, height);
  renderPatches();
  drawMainCanvas();
  els.downloadZip.disabled = state.acceptedPatchIds.size === 0;
  setMode("review");
  setStatus(`Normalized ${width}x${height}; selected ${state.acceptedPatchIds.size} non-overlapping patches.`);
}

function estimateScaleFactor() {
  const markedMedian = median(state.tumorCells.map((cell) => cell.radius * 2));
  const reference = state.librarySummary?.neoplastic_diameter_px_median
    || state.librarySummary?.nucleus_diameter_px_median
    || 25;
  if (!markedMedian) return 1;
  return clamp(reference / markedMedian, 0.4, 8);
}

function renderPatches() {
  els.patchGrid.replaceChildren();
  for (const patch of state.patches) {
    const card = document.createElement("article");
    card.className = `patchCard${patch.accepted ? "" : " rejected"}`;
    const canvas = document.createElement("canvas");
    canvas.width = patch.width;
    canvas.height = patch.height;
    const cardCtx = canvas.getContext("2d");
    if (state.normalized) {
      cardCtx.drawImage(
        state.normalized.imageCanvas,
        patch.x, patch.y, patch.width, patch.height,
        0, 0, patch.width, patch.height
      );
    }
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${patch.editScale} | ${patch.boundaryType} | score ${patch.selectionScore}`;
    const accept = document.createElement("button");
    accept.textContent = patch.accepted ? "Accepted" : "Accept";
    accept.classList.toggle("active", patch.accepted);
    accept.addEventListener("click", () => {
      if (!patch.accepted) {
        const conflict = state.patches.find((item) => (
          item.id !== patch.id
          && state.acceptedPatchIds.has(item.id)
          && patchesOverlap(item, patch)
        ));
        if (conflict) {
          setStatus(`Cannot accept ${patch.id}; it overlaps accepted patch ${conflict.id}.`);
          return;
        }
      }
      patch.accepted = !patch.accepted;
      if (patch.accepted) state.acceptedPatchIds.add(patch.id);
      else state.acceptedPatchIds.delete(patch.id);
      els.downloadZip.disabled = state.acceptedPatchIds.size === 0;
      renderPatches();
      drawMainCanvas();
    });
    const edit = document.createElement("button");
    edit.textContent = "Annotate nuclei";
    edit.addEventListener("click", () => {
      state.selectedPatchId = patch.id;
      setMode("nuclei");
      drawMainCanvas();
    });
    card.append(canvas, meta, accept, edit);
    els.patchGrid.appendChild(card);
  }
}

function drawMainCanvas() {
  ctx.clearRect(0, 0, els.mainCanvas.width, els.mainCanvas.height);
  if (!state.imageBitmap) return;
  if (state.normalized) {
    ctx.drawImage(state.normalized.imageCanvas, 0, 0);
    drawMaskOverlay(ctx, state.normalized.mask, state.normalized.width, state.normalized.height, 0.26);
    drawPatchBoxes();
  } else {
    ctx.drawImage(state.imageBitmap, 0, 0);
    drawMaskOverlay(ctx, state.tissueMask, state.image.width, state.image.height, 0.32);
    drawTumorCells();
  }
  if (state.mode === "nuclei" && state.selectedPatchId) {
    drawPatchNuclei();
  }
}

function drawTumorCells() {
  ctx.save();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.fillStyle = "rgba(217,45,32,0.35)";
  for (const cell of state.tumorCells) {
    ctx.beginPath();
    ctx.arc(cell.x, cell.y, cell.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawPatchBoxes() {
  ctx.save();
  ctx.lineWidth = 3;
  for (const patch of state.patches) {
    ctx.strokeStyle = patch.accepted ? "#0f766e" : "rgba(102,112,133,0.55)";
    ctx.strokeRect(patch.x, patch.y, patch.width, patch.height);
  }
  ctx.restore();
}

function drawPatchNuclei() {
  const patch = state.patches.find((item) => item.id === state.selectedPatchId);
  if (!patch) return;
  ctx.save();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  const nuclei = state.nucleiByPatch.get(patch.id) || [];
  for (const nucleus of nuclei) {
    const color = nucleiLabels.find((item) => item[1] === nucleus.label)?.[2] || "#d92d20";
    ctx.fillStyle = `${color}88`;
    ctx.beginPath();
    ctx.arc(patch.x + nucleus.x, patch.y + nucleus.y, nucleus.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawMaskOverlay(targetCtx, mask, width, height, alpha) {
  if (!mask) return;
  const imageData = targetCtx.getImageData(0, 0, width, height);
  for (let i = 0; i < mask.length; i += 1) {
    const value = mask[i];
    if (value === 0) continue;
    const color = colorForTissue(value);
    const offset = i * 4;
    imageData.data[offset] = imageData.data[offset] * (1 - alpha) + color[0] * alpha;
    imageData.data[offset + 1] = imageData.data[offset + 1] * (1 - alpha) + color[1] * alpha;
    imageData.data[offset + 2] = imageData.data[offset + 2] * (1 - alpha) + color[2] * alpha;
  }
  targetCtx.putImageData(imageData, 0, 0);
}

async function downloadZip() {
  if (!state.normalized) return;
  const files = [];
  const records = [];
  const imageId = sanitize(els.imageId.value || "textbook_image");
  const accepted = state.patches.filter((patch) => state.acceptedPatchIds.has(patch.id));
  const overlap = findOverlappingPair(accepted);
  if (overlap) {
    setStatus(`Cannot export: accepted patches ${overlap[0].id} and ${overlap[1].id} overlap.`);
    return;
  }
  for (const patch of accepted) {
    const sampleId = `${imageId}_py${patch.y}_px${patch.x}`;
    const imageBlob = await canvasToBlob(cropImageCanvas(state.normalized.imageCanvas, patch));
    const tissueBlob = await canvasToBlob(maskToCanvas(state.normalized.mask, state.normalized.width, state.normalized.height, patch, "tissue"));
    const nucleiBlob = await canvasToBlob(nucleiToCanvas(patch));
    files.push(
      { name: `images/${sampleId}.png`, data: new Uint8Array(await imageBlob.arrayBuffer()) },
      { name: `tissue_masks/${sampleId}.png`, data: new Uint8Array(await tissueBlob.arrayBuffer()) },
      { name: `nuclei_masks/${sampleId}.png`, data: new Uint8Array(await nucleiBlob.arrayBuffer()) }
    );
    records.push({
      dataset: "TEXTBOOK",
      image_id: imageId,
      sample_id: sampleId,
      image: `images/${sampleId}.png`,
      tissue_mask: `tissue_masks/${sampleId}.png`,
      nuclei_mask: `nuclei_masks/${sampleId}.png`,
      patch_role: "boundary_invasion",
      edit_scale: patch.editScale,
      boundary_type: patch.boundaryType,
      recommended_edit_type: recommendedEditType(patch.boundaryType),
      change_ratio_target: patch.estimatedChangeRatio,
      source_prompt: "",
      edit_instruction: "",
      target_prompt: "",
      doctor_notes: "",
      estimated_mpp: 0.25,
      scale_factor: state.normalized.scaleFactor,
      scale_source: "doctor_marked_tumor_cells",
      scale_confidence: state.tumorCells.length >= 10 ? "medium" : "low",
      nuclei_library_key: els.libraryKey.value,
      annotation_quality: "accepted",
      artifact_status: "accepted",
      benchmark_group: "textbook_boundary_invasion",
      selection_score: patch.selectionScore,
      tumor_ratio: patch.tumorRatio,
      stroma_ratio: patch.stromaRatio,
      background_ratio: patch.backgroundRatio
    });
  }

  files.push(
    { name: "metadata_textbook_edit.jsonl", data: records.map((row) => JSON.stringify(row)).join("\n") + "\n" },
    { name: "image_level_metadata.jsonl", data: JSON.stringify(imageLevelMetadata(imageId)) + "\n" },
    { name: "manifest.json", data: JSON.stringify({ app_version: "0.1.0", exported_at: new Date().toISOString(), source_image: state.imageName }, null, 2) + "\n" }
  );
  const zip = await buildZip(files);
  const link = document.createElement("a");
  link.href = URL.createObjectURL(zip);
  link.download = `${imageId}.zip`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function imageLevelMetadata(imageId) {
  return {
    source_dataset: "TEXTBOOK",
    image_id: imageId,
    image: state.imageName,
    organ: els.organ.value,
    cancer_type: els.cancerType.value,
    nuclei_library_key: els.libraryKey.value,
    source_mpp: null,
    source_objective: "unknown",
    source_magnification_notes: null,
    scale_bar_available: false,
    image_width_px: state.image?.width ?? null,
    image_height_px: state.image?.height ?? null,
    global_description: els.globalDescription.value,
    annotation_status: "exported"
  };
}

function updateScaleStatus() {
  const reference = state.librarySummary?.neoplastic_diameter_px_median || state.librarySummary?.nucleus_diameter_px_median;
  const marked = median(state.tumorCells.map((cell) => cell.radius * 2));
  const scale = marked ? estimateScaleFactor() : null;
  els.scaleStatus.textContent = `Cells: ${state.tumorCells.length}/10${reference ? ` | ref ${reference.toFixed(1)} px` : ""}${scale ? ` | scale ${scale.toFixed(2)}x` : ""}`;
}

function cropImageCanvas(source, patch) {
  const canvas = document.createElement("canvas");
  canvas.width = patch.width;
  canvas.height = patch.height;
  canvas.getContext("2d").drawImage(source, patch.x, patch.y, patch.width, patch.height, 0, 0, patch.width, patch.height);
  return canvas;
}

function maskToCanvas(mask, width, height, patch) {
  const canvas = document.createElement("canvas");
  canvas.width = patch.width;
  canvas.height = patch.height;
  const imageData = new ImageData(patch.width, patch.height);
  for (let y = 0; y < patch.height; y += 1) {
    for (let x = 0; x < patch.width; x += 1) {
      const sourceX = patch.x + x;
      const sourceY = patch.y + y;
      const value = sourceX >= 0 && sourceY >= 0 && sourceX < width && sourceY < height ? mask[sourceY * width + sourceX] : 0;
      const idx = (y * patch.width + x) * 4;
      imageData.data[idx] = value;
      imageData.data[idx + 1] = value;
      imageData.data[idx + 2] = value;
      imageData.data[idx + 3] = 255;
    }
  }
  canvas.getContext("2d").putImageData(imageData, 0, 0);
  return canvas;
}

function nucleiToCanvas(patch) {
  const canvas = document.createElement("canvas");
  canvas.width = patch.width;
  canvas.height = patch.height;
  const mask = new Uint8Array(patch.width * patch.height);
  const nuclei = state.nucleiByPatch.get(patch.id) || [];
  for (const nucleus of nuclei) {
    paintCircle(mask, patch.width, patch.height, nucleus.x, nucleus.y, nucleus.radius, nucleus.label);
  }
  return maskToCanvas(mask, patch.width, patch.height, { x: 0, y: 0, width: patch.width, height: patch.height });
}

function resizeCanvas(width, height) {
  els.mainCanvas.width = width;
  els.mainCanvas.height = height;
}

function drawScaledImage(bitmap, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0, width, height);
  return canvas;
}

function scaleMaskNearest(mask, srcWidth, srcHeight, dstWidth, dstHeight) {
  const output = new Uint8Array(dstWidth * dstHeight);
  for (let y = 0; y < dstHeight; y += 1) {
    const sourceY = Math.min(srcHeight - 1, Math.floor((y / dstHeight) * srcHeight));
    for (let x = 0; x < dstWidth; x += 1) {
      const sourceX = Math.min(srcWidth - 1, Math.floor((x / dstWidth) * srcWidth));
      output[y * dstWidth + x] = mask[sourceY * srcWidth + sourceX];
    }
  }
  return output;
}

function paintCircle(mask, width, height, cx, cy, radius, value) {
  const r2 = radius * radius;
  for (let y = Math.max(0, cy - radius); y <= Math.min(height - 1, cy + radius); y += 1) {
    for (let x = Math.max(0, cx - radius); x <= Math.min(width - 1, cx + radius); x += 1) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r2) {
        mask[y * width + x] = value;
      }
    }
  }
}

function colorForTissue(value) {
  const found = tissueLabels.find((item) => item[1] === value);
  return hexToRgb(found?.[2] || "#000000");
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [0, 2, 4].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16));
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sanitize(value) {
  return String(value).replace(/[^a-zA-Z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "textbook_image";
}

function recommendedEditType(boundaryType) {
  if (boundaryType === "tumor_stroma") return "tumor_expansion_into_stroma";
  if (boundaryType === "tumor_necrosis") return "necrosis_boundary_expansion";
  if (boundaryType === "normal_stroma") return "gland_crowding_change";
  return "boundary_invasion_edit";
}

function findOverlappingPair(patches) {
  for (let i = 0; i < patches.length; i += 1) {
    for (let j = i + 1; j < patches.length; j += 1) {
      if (patchesOverlap(patches[i], patches[j])) {
        return [patches[i], patches[j]];
      }
    }
  }
  return null;
}

function canvasToBlob(canvas) {
  return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), "image/png"));
}

function setStatus(message) {
  els.status.textContent = message;
}
