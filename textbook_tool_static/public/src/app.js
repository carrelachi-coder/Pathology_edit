import { selectBoundaryPatches, patchesOverlap } from "./patchSelection.js";
import { buildMaskFromPolygons, removeLastPolygonForLabel, rasterizePolygon } from "./tissuePolygons.js";
import { TARGET_TUMOR_CELLS, medianCellDiameter, remainingCellText } from "./cellPolygons.js";
import { acceptanceConflict, nextPendingPatchIndex, queueCounts } from "./patchReview.js";
import { buildZip } from "./zip.js";

const TISSUE_CANVAS_PADDING = 96;

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
  tissuePolygons: [],
  currentPolygon: [],
  tissueLocked: false,
  tumorCells: [],
  currentCellPolygon: [],
  normalized: null,
  patches: [],
  acceptedPatchIds: new Set(),
  selectedPatchId: null,
  nucleiByPatch: new Map(),
  currentPatchIndex: -1,
  currentNucleusPolygon: [],
  mode: "tissue",
  tissueLabel: 1,
  nucleiLabel: 101,
  librarySummary: null,
  zoomEnabled: false,
  viewZoom: 1,
  canvasPadding: 0
};

const els = {};
for (const id of [
  "downloadZip", "imageInput", "metadataInput", "imageId", "organ", "cancerType",
  "libraryKey", "globalDescription", "tissueSection", "tissueLabels", "clearTissue",
  "confirmTissue",
  "cellMode", "clearCells", "scaleStatus", "normalizeAndSelect",
  "nucleiLabels", "patchQueueStatus", "acceptPatch", "rejectPatch", "annotatePatchNuclei",
  "clearPatchNucleus", "confirmPatch", "tissueMode", "zoomMode", "cellToolbarMode",
  "reviewMode", "nucleiMode", "status", "mainCanvas", "canvasWrap", "patchGrid"
]) {
  els[id] = document.getElementById(id);
}

const ctx = els.mainCanvas.getContext("2d", { willReadFrequently: true });

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
  els.clearTissue.addEventListener("click", clearLastTissuePolygon);
  els.confirmTissue.addEventListener("click", confirmTissue);
  els.cellMode.addEventListener("click", () => setMode("cell"));
  els.clearCells.addEventListener("click", () => {
    state.tumorCells.pop();
    updateScaleStatus();
    drawMainCanvas();
  });
  els.normalizeAndSelect.addEventListener("click", normalizeAndSelect);
  els.tissueMode.addEventListener("click", () => setMode("tissue"));
  els.zoomMode.addEventListener("click", toggleZoomMode);
  els.cellToolbarMode.addEventListener("click", () => setMode("cell"));
  els.reviewMode.addEventListener("click", () => setMode("review"));
  els.nucleiMode.addEventListener("click", () => setMode("nuclei"));
  els.acceptPatch.addEventListener("click", acceptCurrentPatch);
  els.rejectPatch.addEventListener("click", rejectCurrentPatch);
  els.annotatePatchNuclei.addEventListener("click", annotateCurrentPatchNuclei);
  els.clearPatchNucleus.addEventListener("click", clearLastPatchNucleus);
  els.confirmPatch.addEventListener("click", confirmCurrentPatch);
  els.downloadZip.addEventListener("click", downloadZip);

  els.mainCanvas.addEventListener("pointerdown", handlePointerDown);
  els.mainCanvas.addEventListener("dblclick", handleCanvasDoubleClick);
  els.canvasWrap.addEventListener("wheel", handleWheelZoom, { passive: false });
  window.addEventListener("keydown", handleKeyDown);
  setPatchControlsDisabled(true);
}

function renderLabelButtons(container, labels, kind) {
  container.replaceChildren();
  for (const [name, value, color] of labels) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${name} (${value})`;
    button.style.borderLeft = `8px solid ${color}`;
    button.disabled = kind === "tissue" && state.tissueLocked;
    if ((kind === "tissue" && value === state.tissueLabel) || (kind === "nuclei" && value === state.nucleiLabel)) {
      button.classList.add("active");
    }
    button.addEventListener("click", () => {
      if (kind === "tissue" && state.tissueLocked) return;
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
  state.tissuePolygons = [];
  state.currentPolygon = [];
  state.tissueLocked = false;
  state.tumorCells = [];
  state.currentCellPolygon = [];
  state.normalized = null;
  state.patches = [];
  state.acceptedPatchIds.clear();
  state.nucleiByPatch.clear();
  state.currentPatchIndex = -1;
  state.currentNucleusPolygon = [];
  state.viewZoom = 1;
  els.imageId.value = file.name.replace(/\.[^.]+$/, "");
  resizeCanvas(state.image.width, state.image.height, TISSUE_CANVAS_PADDING);
  updateTissueLockUI();
  setMode("tissue");
  drawMainCanvas();
  renderPatches();
  updateScaleStatus();
  setStatus(`Loaded ${file.name} (${state.image.width}x${state.image.height}). Default tissue mask is background.`);
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
  if (mode === "tissue" && state.tissueLocked) {
    setStatus("Tissue annotation is locked.");
    return;
  }
  if (mode === "nuclei" && currentPatch()?.status !== "accepted") {
    setStatus("Accept the current patch before annotating nuclei.");
    return;
  }
  state.mode = mode;
  for (const [button, name] of [
    [els.tissueMode, "tissue"],
    [els.cellToolbarMode, "cell"],
    [els.reviewMode, "review"],
    [els.nucleiMode, "nuclei"]
  ]) {
    button.classList.toggle("active", mode === name);
  }
  els.cellMode.classList.toggle("active", mode === "cell");
  updateCanvasCursor();
  drawMainCanvas();
}

function handlePointerDown(event) {
  if (!state.imageBitmap) return;
  const point = canvasPoint(event);
  if (state.mode === "cell") {
    if (event.detail >= 2) {
      completeCurrentCellPolygon();
      return;
    }
    addTumorCellPoint(point);
    return;
  }
  if (state.mode === "nuclei" && state.selectedPatchId) {
    if (event.detail >= 2) {
      completeCurrentPatchNucleus();
      return;
    }
    addPatchNucleusPoint(point);
    return;
  }
  if (state.mode === "tissue") {
    if (state.tissueLocked) return;
    if (event.detail >= 2) {
      completeCurrentPolygon();
      return;
    }
    addTissuePolygonPoint(point);
  }
}

function handleCanvasDoubleClick(event) {
  if (state.mode === "tissue" && !state.tissueLocked) {
    event.preventDefault();
    completeCurrentPolygon();
  }
  if (state.mode === "cell") {
    event.preventDefault();
    completeCurrentCellPolygon();
  }
  if (state.mode === "nuclei" && state.selectedPatchId) {
    event.preventDefault();
    completeCurrentPatchNucleus();
  }
}

function canvasPoint(event) {
  const rect = els.mainCanvas.getBoundingClientRect();
  const offset = imageOffset();
  return {
    x: Math.floor((event.clientX - rect.left) * (els.mainCanvas.width / rect.width) - offset.x),
    y: Math.floor((event.clientY - rect.top) * (els.mainCanvas.height / rect.height) - offset.y)
  };
}

function addTissuePolygonPoint(point) {
  state.currentPolygon.push(point);
  drawMainCanvas();
}

function completeCurrentPolygon() {
  if (state.currentPolygon.length < 3 || !state.imageBitmap) {
    state.currentPolygon = [];
    drawMainCanvas();
    return;
  }
  state.tissuePolygons.push({
    label: state.tissueLabel,
    points: state.currentPolygon.map((point) => ({ ...point }))
  });
  state.currentPolygon = [];
  rebuildTissueMask();
  drawMainCanvas();
}

function clearLastTissuePolygon() {
  if (!state.imageBitmap || state.tissueLocked) return;
  if (state.currentPolygon.length > 0) {
    state.currentPolygon = [];
    drawMainCanvas();
    return;
  }
  state.tissuePolygons = removeLastPolygonForLabel(state.tissuePolygons, state.tissueLabel);
  rebuildTissueMask();
  drawMainCanvas();
}

function confirmTissue() {
  if (!state.imageBitmap || state.tissueLocked) return;
  completeCurrentPolygon();
  state.tissueLocked = true;
  updateTissueLockUI();
  setMode("cell");
  setStatus("Tissue annotation locked. Mark representative tumor cells.");
}

function rebuildTissueMask() {
  if (!state.imageBitmap) return;
  state.tissueMask = buildMaskFromPolygons(state.image.width, state.image.height, state.tissuePolygons);
}

function handleKeyDown(event) {
  if (state.mode === "cell") {
    if (event.key === "Enter") {
      event.preventDefault();
      completeCurrentCellPolygon();
    }
    if (event.key === "Backspace" || event.key === "Delete") {
      event.preventDefault();
      state.currentCellPolygon.pop();
      drawMainCanvas();
    }
    if (event.key === "Escape") {
      state.currentCellPolygon = [];
      drawMainCanvas();
    }
    return;
  }
  if (state.mode === "nuclei" && state.selectedPatchId) {
    if (event.key === "Enter") {
      event.preventDefault();
      completeCurrentPatchNucleus();
    }
    if (event.key === "Backspace" || event.key === "Delete") {
      event.preventDefault();
      state.currentNucleusPolygon.pop();
      drawMainCanvas();
    }
    if (event.key === "Escape") {
      state.currentNucleusPolygon = [];
      drawMainCanvas();
    }
    return;
  }
  if (state.mode !== "tissue" || state.tissueLocked) return;
  if (event.key === "Enter") {
    event.preventDefault();
    completeCurrentPolygon();
  }
  if (event.key === "Backspace" || event.key === "Delete") {
    event.preventDefault();
    state.currentPolygon.pop();
    drawMainCanvas();
  }
  if (event.key === "Escape") {
    state.currentPolygon = [];
    drawMainCanvas();
  }
}

function toggleZoomMode() {
  state.zoomEnabled = !state.zoomEnabled;
  els.zoomMode.classList.toggle("active", state.zoomEnabled);
  els.canvasWrap.classList.toggle("zooming", state.zoomEnabled);
}

function handleWheelZoom(event) {
  if (!state.zoomEnabled || !state.imageBitmap) return;
  event.preventDefault();
  const wrap = els.canvasWrap;
  const wrapRect = wrap.getBoundingClientRect();
  const focalX = event.clientX - wrapRect.left;
  const focalY = event.clientY - wrapRect.top;
  const oldZoom = state.viewZoom;
  const factor = event.deltaY < 0 ? 1.15 : 0.87;
  state.viewZoom = clamp(oldZoom * factor, 0.25, 8);
  applyCanvasZoom();
  const ratio = state.viewZoom / oldZoom;
  wrap.scrollLeft = (wrap.scrollLeft + focalX) * ratio - focalX;
  wrap.scrollTop = (wrap.scrollTop + focalY) * ratio - focalY;
}

function addTumorCellPoint(point) {
  if (!pointInsideImage(point)) {
    setStatus("Tumor cell polygons must be inside the image.");
    return;
  }
  state.currentCellPolygon.push(point);
  drawMainCanvas();
}

function completeCurrentCellPolygon() {
  if (state.currentCellPolygon.length < 3) {
    state.currentCellPolygon = [];
    drawMainCanvas();
    return;
  }
  state.tumorCells.push({
    points: state.currentCellPolygon.map((point) => ({ ...point }))
  });
  state.currentCellPolygon = [];
  updateScaleStatus();
  drawMainCanvas();
}

function addPatchNucleusPoint(point) {
  const patch = currentPatch();
  if (!patch || patch.status !== "accepted") return;
  if (point.x < 0 || point.y < 0 || point.x >= patch.width || point.y >= patch.height) return;
  state.currentNucleusPolygon.push(point);
  drawMainCanvas();
}

function completeCurrentPatchNucleus() {
  const patch = currentPatch();
  if (!patch || patch.status !== "accepted") return;
  if (state.currentNucleusPolygon.length < 3) {
    state.currentNucleusPolygon = [];
    drawMainCanvas();
    return;
  }
  const nuclei = state.nucleiByPatch.get(patch.id) || [];
  nuclei.push({
    label: state.nucleiLabel,
    points: state.currentNucleusPolygon.map((point) => ({ ...point }))
  });
  state.nucleiByPatch.set(patch.id, nuclei);
  state.currentNucleusPolygon = [];
  drawMainCanvas();
}

function setPatchControlsDisabled(disabled) {
  els.acceptPatch.disabled = disabled;
  els.rejectPatch.disabled = disabled;
  els.annotatePatchNuclei.disabled = disabled;
  els.clearPatchNucleus.disabled = disabled;
  els.confirmPatch.disabled = disabled;
}

async function normalizeAndSelect() {
  if (!state.imageBitmap || !state.tissueMask) return;
  const scale = estimateScaleFactor();
  const width = Math.max(1, Math.round(state.image.width * scale));
  const height = Math.max(1, Math.round(state.image.height * scale));
  const imageCanvas = drawScaledImage(state.imageBitmap, width, height);
  const mask = scaleMaskNearest(state.tissueMask, state.image.width, state.image.height, width, height);
  const selection = selectBoundaryPatches(mask, width, height, { patchSize: 512, stride: 256, backupCount: 48 });
  state.normalized = { imageCanvas, mask, width, height, scaleFactor: scale };
  const queue = uniquePatches([...selection.selected, ...selection.backup, ...selection.candidates])
    .sort((a, b) => b.selectionScore - a.selectionScore)
    .slice(0, 48);
  state.patches = queue.map((patch, index) => ({
    ...patch,
    status: "pending",
    rank: index + 1,
    source: selection.selected.some((item) => item.id === patch.id) ? "selected" : "candidate"
  }));
  state.acceptedPatchIds = new Set();
  state.currentPatchIndex = nextPendingPatchIndex(state.patches, -1);
  state.selectedPatchId = state.patches[state.currentPatchIndex]?.id || null;
  state.currentNucleusPolygon = [];
  state.viewZoom = 1;
  const firstPatch = currentPatch();
  resizeCanvas(firstPatch?.width || width, firstPatch?.height || height, 0);
  renderPatches();
  drawMainCanvas();
  els.downloadZip.disabled = true;
  setMode("review");
  updatePatchQueueStatus();
  setStatus(`Normalized ${width}x${height}; opened patch 1 of ${state.patches.length}.`);
}

function estimateScaleFactor() {
  const markedMedian = medianCellDiameter(state.tumorCells);
  const reference = state.librarySummary?.neoplastic_diameter_px_median
    || state.librarySummary?.nucleus_diameter_px_median
    || 25;
  if (!markedMedian) return 1;
  return clamp(reference / markedMedian, 0.4, 8);
}

function uniquePatches(patches) {
  const seen = new Set();
  const unique = [];
  for (const patch of patches) {
    if (seen.has(patch.id)) continue;
    seen.add(patch.id);
    unique.push(patch);
  }
  return unique;
}

function renderPatches() {
  els.patchGrid.replaceChildren();
  for (const [index, patch] of state.patches.entries()) {
    const card = document.createElement("article");
    card.className = `patchCard ${patch.status}`;
    card.classList.toggle("current", index === state.currentPatchIndex);
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
    const stateLabel = document.createElement("span");
    stateLabel.className = "state";
    stateLabel.textContent = `${index + 1}/${state.patches.length} ${patch.status}`;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${patch.editScale} | ${patch.boundaryType} | score ${patch.selectionScore}`;
    const open = document.createElement("button");
    open.textContent = index === state.currentPatchIndex ? "Open" : "Open patch";
    open.classList.toggle("active", index === state.currentPatchIndex);
    open.disabled = patch.status === "confirmed";
    open.addEventListener("click", () => {
      openPatchAt(index);
    });
    card.append(canvas, stateLabel, meta, open);
    els.patchGrid.appendChild(card);
  }
}

function currentPatch() {
  return state.currentPatchIndex >= 0 ? state.patches[state.currentPatchIndex] : null;
}

function openPatchAt(index) {
  const patch = state.patches[index];
  if (!patch || patch.status === "confirmed") return;
  state.currentPatchIndex = index;
  state.selectedPatchId = patch.id;
  state.currentNucleusPolygon = [];
  resizeCanvas(patch.width, patch.height, 0);
  setMode(patch.status === "accepted" ? "nuclei" : "review");
  updatePatchQueueStatus();
  renderPatches();
  drawMainCanvas();
}

function acceptCurrentPatch() {
  const patch = currentPatch();
  if (!patch || patch.status !== "pending") return;
  const conflict = acceptanceConflict(patch, state.patches);
  if (conflict) {
    setStatus(`Cannot accept patch ${patch.rank}; it overlaps confirmed patch ${conflict.rank}.`);
    return;
  }
  patch.status = "accepted";
  state.selectedPatchId = patch.id;
  setMode("nuclei");
  updatePatchQueueStatus();
  renderPatches();
  drawMainCanvas();
}

function rejectCurrentPatch() {
  const patch = currentPatch();
  if (!patch || patch.status === "confirmed") return;
  patch.status = "rejected";
  state.currentNucleusPolygon = [];
  state.nucleiByPatch.delete(patch.id);
  advanceToNextPatch();
}

function annotateCurrentPatchNuclei() {
  const patch = currentPatch();
  if (!patch) return;
  if (patch.status !== "accepted") {
    setStatus("Accept this patch before annotating nuclei.");
    return;
  }
  setMode("nuclei");
}

function clearLastPatchNucleus() {
  const patch = currentPatch();
  if (!patch || patch.status !== "accepted") return;
  const nuclei = state.nucleiByPatch.get(patch.id) || [];
  nuclei.pop();
  state.nucleiByPatch.set(patch.id, nuclei);
  drawMainCanvas();
}

function confirmCurrentPatch() {
  const patch = currentPatch();
  if (!patch || patch.status !== "accepted") return;
  const nuclei = state.nucleiByPatch.get(patch.id) || [];
  if (nuclei.length === 0) {
    setStatus("Annotate nuclei before confirming this patch.");
    return;
  }
  patch.status = "confirmed";
  state.currentNucleusPolygon = [];
  state.acceptedPatchIds.add(patch.id);
  els.downloadZip.disabled = state.acceptedPatchIds.size === 0;
  advanceToNextPatch();
}

function advanceToNextPatch() {
  const nextIndex = nextPendingPatchIndex(state.patches, state.currentPatchIndex);
  if (nextIndex >= 0) {
    openPatchAt(nextIndex);
    setStatus(`Opened patch ${nextIndex + 1} of ${state.patches.length}.`);
    return;
  }
  state.currentPatchIndex = -1;
  state.selectedPatchId = null;
  if (state.normalized) {
    resizeCanvas(state.normalized.width, state.normalized.height, 0);
  }
  setMode("review");
  updatePatchQueueStatus();
  renderPatches();
  drawMainCanvas();
  setStatus("Patch queue complete. Download zip when ready.");
}

function updatePatchQueueStatus() {
  const counts = queueCounts(state.patches);
  const current = currentPatch();
  setPatchControlsDisabled(!current);
  if (current) {
    els.acceptPatch.disabled = current.status !== "pending";
    els.rejectPatch.disabled = current.status === "confirmed";
    els.annotatePatchNuclei.disabled = current.status !== "accepted";
    els.clearPatchNucleus.disabled = current.status !== "accepted";
    els.confirmPatch.disabled = current.status !== "accepted";
  }
  els.patchQueueStatus.textContent = state.patches.length
    ? `Queue: ${counts.confirmed} confirmed, ${counts.rejected} rejected, ${counts.pending} pending${current ? ` | current ${state.currentPatchIndex + 1}/${state.patches.length}: ${current.status}` : ""}`
    : "Normalize and select patches first.";
}

function drawMainCanvas() {
  ctx.clearRect(0, 0, els.mainCanvas.width, els.mainCanvas.height);
  if (!state.imageBitmap) return;
  if (state.normalized) {
    const patch = currentPatch();
    if (patch) {
      ctx.drawImage(
        state.normalized.imageCanvas,
        patch.x, patch.y, patch.width, patch.height,
        0, 0, patch.width, patch.height
      );
      drawPatchNuclei();
    } else {
      ctx.drawImage(state.normalized.imageCanvas, 0, 0);
      drawMaskOverlay(ctx, state.normalized.mask, state.normalized.width, state.normalized.height, 0.26);
      drawPatchBoxes();
    }
  } else {
    const offset = imageOffset();
    ctx.fillStyle = "#eef1f5";
    ctx.fillRect(0, 0, els.mainCanvas.width, els.mainCanvas.height);
    ctx.drawImage(state.imageBitmap, offset.x, offset.y);
    drawMaskOverlay(ctx, state.tissueMask, state.image.width, state.image.height, 0.32, offset.x, offset.y, 0.28);
    drawImageBoundary(offset);
    drawCurrentPolygon();
    drawTumorCells();
  }
}

function drawTumorCells() {
  ctx.save();
  ctx.lineWidth = 2;
  const offset = imageOffset();
  for (const [index, cell] of state.tumorCells.entries()) {
    drawPolygonPath(cell.points, offset);
    ctx.fillStyle = "rgba(217,45,32,0.36)";
    ctx.strokeStyle = "#ffffff";
    ctx.fill();
    ctx.stroke();
    const center = polygonCentroid(cell.points);
    ctx.fillStyle = "#ffffff";
    ctx.font = "12px Arial";
    ctx.fillText(String(index + 1), offset.x + center.x - 4, offset.y + center.y + 4);
  }
  if (state.mode === "cell" && state.currentCellPolygon.length > 0) {
    drawPolygonPath(state.currentCellPolygon, offset);
    ctx.strokeStyle = "#d92d20";
    ctx.fillStyle = "rgba(217,45,32,0.18)";
    if (state.currentCellPolygon.length >= 3) {
      ctx.fill();
    }
    ctx.stroke();
  }
  for (const point of state.currentCellPolygon) {
    ctx.beginPath();
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#d92d20";
    ctx.arc(offset.x + point.x, offset.y + point.y, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawCurrentPolygon() {
  if (state.mode !== "tissue" || state.currentPolygon.length === 0) return;
  const color = tissueLabels.find((item) => item[1] === state.tissueLabel)?.[2] || "#0f766e";
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = `${color}33`;
  ctx.lineWidth = 2;
  const offset = imageOffset();
  drawPolygonPath(state.currentPolygon, offset);
  if (state.currentPolygon.length >= 3) {
    ctx.fill();
  }
  ctx.stroke();
  ctx.fillStyle = "#ffffff";
  for (const point of state.currentPolygon) {
    ctx.beginPath();
    ctx.arc(offset.x + point.x, offset.y + point.y, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawPolygonPath(points, offset) {
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = offset.x + point.x;
    const y = offset.y + point.y;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  if (points.length >= 3) {
    ctx.closePath();
  }
}

function polygonCentroid(points) {
  if (!points.length) return { x: 0, y: 0 };
  const sum = points.reduce((acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }), { x: 0, y: 0 });
  return { x: sum.x / points.length, y: sum.y / points.length };
}

function drawPatchBoxes() {
  ctx.save();
  ctx.lineWidth = 3;
  for (const patch of state.patches) {
    ctx.strokeStyle = patch.status === "confirmed" ? "#0f766e" : "rgba(102,112,133,0.55)";
    ctx.strokeRect(patch.x, patch.y, patch.width, patch.height);
  }
  ctx.restore();
}

function drawPatchNuclei() {
  const patch = currentPatch();
  if (!patch) return;
  ctx.save();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  const nuclei = state.nucleiByPatch.get(patch.id) || [];
  for (const nucleus of nuclei) {
    const color = nucleiLabels.find((item) => item[1] === nucleus.label)?.[2] || "#d92d20";
    drawPolygonPath(nucleus.points, { x: 0, y: 0 });
    ctx.fillStyle = `${color}66`;
    ctx.fill();
    ctx.stroke();
  }
  if (state.mode === "nuclei" && state.currentNucleusPolygon.length > 0) {
    const color = nucleiLabels.find((item) => item[1] === state.nucleiLabel)?.[2] || "#d92d20";
    drawPolygonPath(state.currentNucleusPolygon, { x: 0, y: 0 });
    ctx.strokeStyle = color;
    ctx.fillStyle = `${color}22`;
    if (state.currentNucleusPolygon.length >= 3) ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#fff";
    for (const point of state.currentNucleusPolygon) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawImageBoundary(offset) {
  ctx.save();
  ctx.strokeStyle = "rgba(24,32,42,0.55)";
  ctx.setLineDash([8, 6]);
  ctx.lineWidth = 1.5;
  ctx.strokeRect(offset.x, offset.y, state.image.width, state.image.height);
  ctx.restore();
}

function drawMaskOverlay(targetCtx, mask, width, height, alpha, offsetX = 0, offsetY = 0, backgroundAlpha = 0) {
  if (!mask) return;
  const imageData = targetCtx.getImageData(offsetX, offsetY, width, height);
  for (let i = 0; i < mask.length; i += 1) {
    const value = mask[i];
    if (value === 0 && backgroundAlpha <= 0) continue;
    const blendAlpha = value === 0 ? backgroundAlpha : alpha;
    const color = value === 0 ? [60, 72, 88] : colorForTissue(value);
    const offset = i * 4;
    imageData.data[offset] = imageData.data[offset] * (1 - blendAlpha) + color[0] * blendAlpha;
    imageData.data[offset + 1] = imageData.data[offset + 1] * (1 - blendAlpha) + color[1] * blendAlpha;
    imageData.data[offset + 2] = imageData.data[offset + 2] * (1 - blendAlpha) + color[2] * blendAlpha;
  }
  targetCtx.putImageData(imageData, offsetX, offsetY);
}

async function downloadZip() {
  if (!state.normalized) return;
  const files = [];
  const records = [];
  const imageId = sanitize(els.imageId.value || "textbook_image");
  const accepted = state.patches.filter((patch) => patch.status === "confirmed");
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
      scale_confidence: state.tumorCells.length >= TARGET_TUMOR_CELLS ? "medium" : "low",
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
  const marked = medianCellDiameter(state.tumorCells);
  const scale = marked ? estimateScaleFactor() : null;
  const inProgress = state.currentCellPolygon.length > 0 ? " | drawing cell" : "";
  els.scaleStatus.textContent = `${remainingCellText(state.tumorCells.length)}${inProgress}${reference ? ` | ref ${reference.toFixed(1)} px` : ""}${marked ? ` | median ${marked.toFixed(1)} px` : ""}${scale ? ` | scale ${scale.toFixed(2)}x` : ""}`;
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
    rasterizePolygon(mask, patch.width, patch.height, nucleus.points, nucleus.label);
  }
  return maskToCanvas(mask, patch.width, patch.height, { x: 0, y: 0, width: patch.width, height: patch.height });
}

function resizeCanvas(width, height, padding = 0) {
  state.canvasPadding = padding;
  els.mainCanvas.width = width + padding * 2;
  els.mainCanvas.height = height + padding * 2;
  applyCanvasZoom();
}

function applyCanvasZoom() {
  els.mainCanvas.style.width = `${els.mainCanvas.width * state.viewZoom}px`;
  els.mainCanvas.style.height = `${els.mainCanvas.height * state.viewZoom}px`;
}

function updateCanvasCursor() {
  els.mainCanvas.classList.toggle("polygonMode", state.mode === "tissue" && !state.tissueLocked);
  els.mainCanvas.classList.toggle("cellPolygonMode", state.mode === "cell");
}

function updateTissueLockUI() {
  els.tissueSection.classList.toggle("locked", state.tissueLocked);
  els.clearTissue.disabled = state.tissueLocked;
  els.confirmTissue.disabled = state.tissueLocked;
  els.tissueMode.disabled = state.tissueLocked;
  renderLabelButtons(els.tissueLabels, tissueLabels, "tissue");
  updateCanvasCursor();
}

function imageOffset() {
  return state.normalized ? { x: 0, y: 0 } : { x: state.canvasPadding, y: state.canvasPadding };
}

function pointInsideImage(point) {
  const width = state.normalized?.width ?? state.image?.width ?? 0;
  const height = state.normalized?.height ?? state.image?.height ?? 0;
  return point.x >= 0 && point.y >= 0 && point.x < width && point.y < height;
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

function colorForTissue(value) {
  const found = tissueLabels.find((item) => item[1] === value);
  return hexToRgb(found?.[2] || "#000000");
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [0, 2, 4].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16));
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
