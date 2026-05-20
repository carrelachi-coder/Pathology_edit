import { buildMaskFromPolygons, removeLastPolygonForLabel } from "./tissuePolygons.js";
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

const state = {
  batchFiles: [],
  completedMasks: new Map(),
  currentIndex: -1,
  image: null,
  imageBitmap: null,
  imageName: "",
  imageId: "textbook_image",
  tissueMask: null,
  tissuePolygons: [],
  currentPolygon: [],
  tissueLocked: false,
  baseLabel: 0,
  tissueLabel: 1,
  zoomEnabled: false,
  viewZoom: 1,
  canvasPadding: 0
};

const els = {};
for (const id of [
  "downloadZip",
  "imageInput",
  "folderInput",
  "imageSelector",
  "imageId",
  "baseLabel",
  "tissueSection",
  "tissueLabels",
  "clearTissue",
  "confirmTissue",
  "tissueMode",
  "zoomMode",
  "status",
  "mainCanvas",
  "previewCanvas",
  "canvasWrap"
]) {
  els[id] = document.getElementById(id);
}

const ctx = els.mainCanvas.getContext("2d", { willReadFrequently: true });
const previewCtx = els.previewCanvas.getContext("2d", { willReadFrequently: true });

init();

function init() {
  renderLabelButtons();
  bindEvents();
  refreshImageSelector();
  setStatus("Load a batch folder or single image to begin.");
}

function bindEvents() {
  els.imageInput.addEventListener("change", handleSingleImageInput);
  els.folderInput.addEventListener("change", handleFolderInput);
  els.imageSelector.addEventListener("change", handleImageSelectionChange);
  els.imageId.addEventListener("input", () => {
    state.imageId = sanitize(els.imageId.value);
  });
  els.baseLabel.addEventListener("change", () => {
    state.baseLabel = Number(els.baseLabel.value);
    if (state.imageBitmap) {
      rebuildTissueMask();
      drawMainCanvas();
      setStatus(`Base tissue set to ${currentLabelName(state.baseLabel)}.`);
    }
  });
  els.clearTissue.addEventListener("click", clearLastTissuePolygon);
  els.confirmTissue.addEventListener("click", confirmTissue);
  els.tissueMode.addEventListener("click", () => setMode("tissue"));
  els.zoomMode.addEventListener("click", toggleZoomMode);
  els.downloadZip.addEventListener("click", downloadZip);
  els.mainCanvas.addEventListener("pointerdown", handlePointerDown);
  els.mainCanvas.addEventListener("dblclick", handleCanvasDoubleClick);
  els.canvasWrap.addEventListener("wheel", handleWheelZoom, { passive: false });
  window.addEventListener("keydown", handleKeyDown);
  updateTissueLockUI();
}

function renderLabelButtons() {
  els.tissueLabels.replaceChildren();
  for (const [name, value, color] of tissueLabels) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${name} (${value})`;
    button.style.borderLeft = `8px solid ${color}`;
    button.disabled = state.tissueLocked;
    if (value === state.tissueLabel) button.classList.add("active");
    button.addEventListener("click", () => {
      if (state.tissueLocked) return;
      state.tissueLabel = value;
      renderLabelButtons();
    });
    els.tissueLabels.appendChild(button);
  }
}

async function handleSingleImageInput(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  state.batchFiles = [file];
  state.completedMasks.clear();
  refreshImageSelector();
  await loadBatchIndex(0);
}

async function handleFolderInput(event) {
  const files = [...(event.target.files || [])].filter((file) => file.type.startsWith("image/"));
  if (!files.length) return;
  state.batchFiles = sortFiles(files);
  state.completedMasks.clear();
  refreshImageSelector();
  await loadBatchIndex(0);
  setStatus(`Loaded batch with ${state.batchFiles.length} images. Recommended batch size: up to 50 images to keep browser memory and review flow comfortable.`);
}

function handleImageSelectionChange() {
  const index = Number(els.imageSelector.value);
  if (Number.isNaN(index) || index < 0 || index >= state.batchFiles.length) return;
  loadBatchIndex(index);
}

async function loadBatchIndex(index) {
  if (index < 0 || index >= state.batchFiles.length) return;
  state.currentIndex = index;
  await loadImageFile(state.batchFiles[index], { keepBatch: true });
  els.imageSelector.value = String(index);
  updateImageSelectorOptions();
}

async function loadImageFile(file, options = {}) {
  state.imageName = file.name;
  state.imageId = file.name.replace(/\.[^.]+$/, "");
  state.imageBitmap = await createImageBitmap(file);
  state.image = state.imageBitmap;
  state.tissueMask = new Uint8Array(state.image.width * state.image.height);
  state.tissueMask.fill(state.baseLabel);
  state.tissuePolygons = [];
  state.currentPolygon = [];
  state.tissueLocked = false;
  state.viewZoom = 1;
  els.imageId.value = state.imageId;
  els.baseLabel.value = String(state.baseLabel);
  resizeCanvas(state.image.width, state.image.height, 96);
  resizePreviewCanvas(state.image.width, state.image.height);
  els.downloadZip.disabled = state.batchFiles.length === 0;
  updateTissueLockUI();
  setMode("tissue");
  drawMainCanvas();
  if (!options.keepBatch) {
    refreshImageSelector();
  }
  setStatus(`Loaded ${file.name} (${state.image.width}x${state.image.height}). Base tissue is ${currentLabelName(state.baseLabel)}.`);
}

function refreshImageSelector() {
  els.imageSelector.replaceChildren();
  if (!state.batchFiles.length) {
    const option = document.createElement("option");
    option.value = "-1";
    option.textContent = "No batch loaded";
    els.imageSelector.appendChild(option);
    els.imageSelector.disabled = true;
    els.downloadZip.disabled = true;
    return;
  }

  els.imageSelector.disabled = false;
  updateImageSelectorOptions();
}

function updateImageSelectorOptions() {
  if (!state.batchFiles.length) return;
  els.imageSelector.replaceChildren();
  const pending = [];
  const done = [];
  state.batchFiles.forEach((file, index) => {
    const bucket = state.completedMasks.has(file.name) ? done : pending;
    bucket.push({ file, index });
  });
  const groups = [
    ["Pending", pending],
    ["Completed", done]
  ];
  for (const [label, items] of groups) {
    if (!items.length) continue;
    const group = document.createElement("optgroup");
    group.label = `${label} (${items.length})`;
    for (const { file, index } of items) {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = file.name;
      group.appendChild(option);
    }
    els.imageSelector.appendChild(group);
  }
  if (state.currentIndex >= 0) {
    els.imageSelector.value = String(state.currentIndex);
  }
}

function handlePointerDown(event) {
  if (!state.imageBitmap || state.tissueLocked || state.zoomEnabled) return;
  const point = canvasPoint(event);
  if (event.detail >= 2) {
    completeCurrentPolygon();
    return;
  }
  addTissuePolygonPoint(point);
}

function handleCanvasDoubleClick(event) {
  if (!state.imageBitmap || state.tissueLocked || state.zoomEnabled) return;
  event.preventDefault();
  completeCurrentPolygon();
}

function addTissuePolygonPoint(point) {
  state.currentPolygon.push(point);
  drawMainCanvas();
  setStatus(`Tissue points: ${state.currentPolygon.length}. Double-click or press Enter to close.`);
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
  setStatus(`Closed tissue polygon #${state.tissuePolygons.length}.`);
}

function clearLastTissuePolygon() {
  if (!state.imageBitmap || state.tissueLocked) return;
  if (state.currentPolygon.length > 0) {
    state.currentPolygon = [];
    drawMainCanvas();
    setStatus("Cleared current tissue polygon.");
    return;
  }
  state.tissuePolygons = removeLastPolygonForLabel(state.tissuePolygons, state.tissueLabel);
  rebuildTissueMask();
  drawMainCanvas();
  setStatus(`Removed last ${currentLabelName(state.tissueLabel)} polygon.`);
}

function confirmTissue() {
  if (!state.imageBitmap || state.tissueLocked) return;
  completeCurrentPolygon();
  state.tissueLocked = true;
  state.completedMasks.set(state.imageName, cloneMask(state.tissueMask));
  updateTissueLockUI();
  setStatus(`Tissue annotation confirmed for ${state.imageName}.`);
  if (state.currentIndex + 1 < state.batchFiles.length) {
    loadBatchIndex(state.currentIndex + 1);
  } else {
    setStatus(`Tissue annotation confirmed for ${state.imageName}. Batch complete.`);
  }
}

function rebuildTissueMask() {
  if (!state.imageBitmap) return;
  state.tissueMask = buildMaskFromPolygons(state.image.width, state.image.height, state.tissuePolygons, state.baseLabel);
}

function handleKeyDown(event) {
  if (!state.imageBitmap || state.tissueLocked) return;
  if (event.key === "Enter") {
    event.preventDefault();
    completeCurrentPolygon();
  }
  if (event.key === "Backspace" || event.key === "Delete") {
    event.preventDefault();
    state.currentPolygon.pop();
    drawMainCanvas();
    setStatus(`Tissue points: ${state.currentPolygon.length}.`);
  }
  if (event.key === "Escape") {
    state.currentPolygon = [];
    drawMainCanvas();
    setStatus("Cleared current tissue polygon.");
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

function setMode(mode) {
  if (mode === "tissue" && state.tissueLocked) return;
  els.tissueMode.classList.toggle("active", mode === "tissue");
  drawMainCanvas();
}

function updateTissueLockUI() {
  els.tissueSection.classList.toggle("locked", state.tissueLocked);
  els.clearTissue.disabled = state.tissueLocked;
  els.confirmTissue.disabled = state.tissueLocked;
  els.tissueMode.disabled = state.tissueLocked;
  renderLabelButtons();
  updateImageSelectorOptions();
}

function drawMainCanvas() {
  if (!state.imageBitmap) {
    ctx.clearRect(0, 0, els.mainCanvas.width, els.mainCanvas.height);
    return;
  }
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, els.mainCanvas.width, els.mainCanvas.height);
  ctx.restore();

  const offset = imageOffset();
  ctx.drawImage(state.imageBitmap, offset.x, offset.y);
  drawTissueOverlay(offset);
  drawCurrentPolygon(offset);
  drawPreview();
}

function drawTissueOverlay(offset) {
  if (!state.tissueMask || !state.imageBitmap) return;
  const width = state.image.width;
  const height = state.image.height;
  const imageData = ctx.getImageData(offset.x, offset.y, width, height);
  for (let i = 0; i < state.tissueMask.length; i += 1) {
    const value = state.tissueMask[i];
    if (value === 0) continue;
    const [r, g, b] = hexToRgb(colorForTissue(value));
    const idx = i * 4;
    imageData.data[idx] = Math.round(imageData.data[idx] * 0.28 + r * 0.72);
    imageData.data[idx + 1] = Math.round(imageData.data[idx + 1] * 0.28 + g * 0.72);
    imageData.data[idx + 2] = Math.round(imageData.data[idx + 2] * 0.28 + b * 0.72);
  }
  ctx.putImageData(imageData, offset.x, offset.y);
}

function drawCurrentPolygon(offset) {
  if (state.currentPolygon.length === 0) return;
  ctx.save();
  ctx.strokeStyle = colorForTissue(state.tissueLabel);
  ctx.fillStyle = "rgba(15, 118, 110, 0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [index, point] of state.currentPolygon.entries()) {
    const x = point.x + offset.x;
    const y = point.y + offset.y;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  for (const point of state.currentPolygon) {
    const x = point.x + offset.x;
    const y = point.y + offset.y;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function canvasPoint(event) {
  const rect = els.mainCanvas.getBoundingClientRect();
  const offset = imageOffset();
  return {
    x: Math.floor((event.clientX - rect.left) * (els.mainCanvas.width / rect.width) - offset.x),
    y: Math.floor((event.clientY - rect.top) * (els.mainCanvas.height / rect.height) - offset.y)
  };
}

function imageOffset() {
  return { x: state.canvasPadding, y: state.canvasPadding };
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

function resizePreviewCanvas(width, height) {
  els.previewCanvas.width = width;
  els.previewCanvas.height = height;
}

function drawPreview() {
  if (!state.imageBitmap) {
    previewCtx.clearRect(0, 0, els.previewCanvas.width, els.previewCanvas.height);
    return;
  }
  previewCtx.clearRect(0, 0, els.previewCanvas.width, els.previewCanvas.height);
  previewCtx.drawImage(state.imageBitmap, 0, 0, els.previewCanvas.width, els.previewCanvas.height);
}

async function downloadZip() {
  if (!state.batchFiles.length) return;
  if (state.imageBitmap && !state.tissueLocked) {
    completeCurrentPolygon();
    state.completedMasks.set(state.imageName, cloneMask(state.tissueMask));
  }
  const files = [];
  for (const file of state.batchFiles) {
    const mask = state.completedMasks.get(file.name);
    if (!mask) continue;
    const imageId = sanitize(file.name.replace(/\.[^.]+$/, ""));
    files.push({
      name: `masks/${imageId}_mask.png`,
      data: new Uint8Array(await canvasToBlob(maskToCanvas(mask, mask.width, mask.height)).then((blob) => blob.arrayBuffer()))
    });
  }
  if (!files.length) {
    setStatus("No completed masks to download yet.");
    return;
  }
  const zip = await buildZip(files);
  const link = document.createElement("a");
  link.href = URL.createObjectURL(zip);
  link.download = `tissue_masks_batch.zip`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function maskToCanvas(mask, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const imageData = new ImageData(width, height);
  for (let i = 0; i < mask.length; i += 1) {
    const value = mask[i];
    const idx = i * 4;
    imageData.data[idx] = value;
    imageData.data[idx + 1] = value;
    imageData.data[idx + 2] = value;
    imageData.data[idx + 3] = 255;
  }
  canvas.getContext("2d").putImageData(imageData, 0, 0);
  return canvas;
}

function canvasToBlob(canvas) {
  return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), "image/png"));
}

function cloneMask(mask) {
  const copy = new Uint8Array(mask.length);
  copy.set(mask);
  return copy;
}

function colorForTissue(value) {
  return tissueLabels.find((item) => item[1] === value)?.[2] || "#000000";
}

function currentLabelName(value) {
  return tissueLabels.find((item) => item[1] === value)?.[0] || "background";
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [0, 2, 4].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16));
}

function sortFiles(files) {
  return [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sanitize(value) {
  return String(value).replace(/[^a-zA-Z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "textbook_image";
}

function setStatus(message) {
  els.status.textContent = message;
}
