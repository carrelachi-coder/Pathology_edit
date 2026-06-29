import { buildMaskFromPolygons, removeLastPolygonForLabel } from "./tissuePolygons.js";
import { buildZip } from "./zip.js";
import {
  buildReviewItems,
  emptyManifest as emptySelectionManifest,
  getSelectionMetadataForFile,
  parseSelectionManifest,
  summarizeReviewItems
} from "./maskReview.js";

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

const AUTO_DOWNLOAD_THRESHOLD = 10;
const STORAGE_KEY = "pathology_annotation_progress";
const STORAGE_BATCH_KEY = "pathology_annotation_batch_id";

const state = {
  batchFiles: [],
  batchManifest: emptySelectionManifest(),
  completedMasks: new Map(),
  downloadedNames: new Set(),
  batchId: "",
  masksSinceDownload: 0,
  downloadBlocked: false,
  currentMetadata: null,
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
  canvasPadding: 0,
  mode: "annotate",
  reviewItems: [],
  reviewIndex: -1,
  reviewMask: null,
  reviewOpacity: 0.45,
  reviewShowMissing: true
};

const els = {};
for (const id of [
  "downloadZip",
  "skipToPending",
  "progressBadge",
  "downloadBanner",
  "imageInput",
  "folderInput",
  "reviewMaskInput",
  "reviewSummary",
  "imageSelector",
  "imageId",
  "selectionMetadata",
  "selectionOrgan",
  "selectionCaption",
  "baseLabel",
  "tissueSection",
  "reviewSection",
  "reviewOpacity",
  "reviewShowMissing",
  "reviewLabels",
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
  els.imageInput.addEventListener("click", clearFileInput);
  els.imageInput.addEventListener("change", handleSingleImageInput);
  els.folderInput.addEventListener("click", clearFileInput);
  els.folderInput.addEventListener("change", handleFolderInput);
  els.reviewMaskInput.addEventListener("click", clearFileInput);
  els.reviewMaskInput.addEventListener("change", handleReviewMaskInput);
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
  els.reviewOpacity.addEventListener("input", () => {
    state.reviewOpacity = Number(els.reviewOpacity.value) / 100;
    drawMainCanvas();
  });
  els.reviewShowMissing.addEventListener("change", () => {
    state.reviewShowMissing = els.reviewShowMissing.checked;
    updateImageSelectorOptions();
  });
  els.downloadZip.addEventListener("click", handleManualDownload);
  els.skipToPending.addEventListener("click", skipToNextPending);
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
  try {
    const file = event.target.files?.[0];
    if (!file) return;
    enterAnnotationMode();
    setStatus(`Loading ${file.name}...`);
    state.batchFiles = [file];
    state.batchManifest = emptySelectionManifest();
    state.completedMasks.clear();
    refreshImageSelector();
    await loadBatchIndex(0);
  } catch (error) {
    setStatus(`Could not load image: ${formatErrorMessage(error)}`);
  }
}

async function handleFolderInput(event) {
  try {
    const selectedFiles = [...(event.target.files || [])];
    if (!selectedFiles.length) {
      setStatus("No files were received from the folder picker. Try choosing the folder again, or hard-refresh this page first.");
      return;
    }
    const files = selectedFiles.filter(isImageFile);
    const csvCount = selectedFiles.filter(isCsvFile).length;
    if (!files.length) {
      const sampleNames = selectedFiles.slice(0, 5).map((file) => file.name || "(unnamed)").join(", ");
      setStatus(`Folder picker returned ${selectedFiles.length} files, but no PNG/image files were recognized. First files: ${sampleNames}`);
      return;
    }
    enterAnnotationMode();
    setStatus(`Loading batch folder: ${selectedFiles.length} files selected, ${files.length} images, ${csvCount} CSV files...`);
    const manifestFile = pickSelectionManifestFile(selectedFiles);
    state.batchManifest = manifestFile
      ? parseSelectionManifest(await readFileAsText(manifestFile))
      : emptySelectionManifest();
    state.batchFiles = sortFiles(files);
    state.batchId = computeBatchId(state.batchFiles);
    loadProgress();
    state.completedMasks.clear();
    refreshImageSelector();
    updateProgressBadge();
    const firstPending = state.batchFiles.findIndex(
      (f) => !state.downloadedNames.has(f.name)
    );
    await loadBatchIndex(firstPending >= 0 ? firstPending : 0);
    const matchedRows = state.batchFiles.filter((file) => getSelectionMetadataForFile(state.batchManifest, file)).length;
    const manifestStatus = manifestFile
      ? ` Matched ${matchedRows}/${state.batchFiles.length} images to ${manifestFile.name}.`
      : " No CSV manifest found in this folder.";
    const skippedCount = state.batchFiles.filter((f) => state.downloadedNames.has(f.name)).length;
    const resumeMsg = skippedCount > 0 ? ` Resuming: ${skippedCount} already downloaded, starting at first pending.` : "";
    setStatus(`Loaded batch with ${state.batchFiles.length} images.${manifestStatus}${resumeMsg}`);
  } catch (error) {
    setStatus(`Could not load batch folder: ${formatErrorMessage(error)}`);
  }
}

async function handleReviewMaskInput(event) {
  try {
    const selectedFiles = [...(event.target.files || [])];
    if (!selectedFiles.length) {
      setStatus("No mask files were received from the review folder picker.");
      return;
    }
    if (!state.batchFiles.length) {
      setStatus("Load the original image/CSV batch folder first, then choose the review masks folder.");
      return;
    }
    const maskFiles = selectedFiles.filter(isImageFile);
    if (!maskFiles.length) {
      setStatus("The review folder does not contain any image mask files.");
      return;
    }
    enterReviewMode(maskFiles);
    const firstMatched = state.reviewItems.findIndex((item) => item.idMask);
    await loadReviewIndex(firstMatched >= 0 ? firstMatched : 0);
    const summary = summarizeReviewItems(state.reviewItems);
    setStatus(`Review loaded: ${summary.matched}/${summary.total} images have ID masks. Use Image in batch to inspect results.`);
  } catch (error) {
    setStatus(`Could not load review masks: ${formatErrorMessage(error)}`);
  }
}

function enterAnnotationMode() {
  state.mode = "annotate";
  state.reviewItems = [];
  state.reviewIndex = -1;
  state.reviewMask = null;
  if (els.reviewSummary) els.reviewSummary.hidden = true;
  updateModeUI();
}

function enterReviewMode(maskFiles) {
  state.mode = "review";
  state.reviewItems = buildReviewItems(state.batchFiles, maskFiles, state.batchManifest);
  state.reviewIndex = -1;
  state.reviewMask = null;
  state.tissueLocked = true;
  state.currentPolygon = [];
  updateModeUI();
  renderReviewSummary();
  updateImageSelectorOptions();
}

function handleImageSelectionChange() {
  const index = Number(els.imageSelector.value);
  if (state.mode === "review") {
    if (Number.isNaN(index) || index < 0 || index >= state.reviewItems.length) return;
    loadReviewIndex(index);
    return;
  }
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
  state.currentMetadata = getSelectionMetadataForFile(state.batchManifest, file);
  state.imageBitmap = await loadDrawableImage(file);
  state.image = state.imageBitmap;
  state.tissueMask = new Uint8Array(state.image.width * state.image.height);
  state.tissueMask.fill(state.baseLabel);
  state.tissuePolygons = [];
  state.currentPolygon = [];
  state.tissueLocked = state.downloadedNames.has(file.name);
  state.viewZoom = 1;
  els.imageId.value = state.imageId;
  els.baseLabel.value = String(state.baseLabel);
  renderSelectionMetadata();
  resizeCanvas(state.image.width, state.image.height, 96);
  resizePreviewCanvas(state.image.width, state.image.height);
  els.downloadZip.disabled = state.batchFiles.length === 0;
  updateTissueLockUI();
  setMode("tissue");
  drawMainCanvas();
  if (!options.keepBatch) {
    refreshImageSelector();
  }
  if (state.downloadedNames.has(file.name)) {
    setStatus(`Loaded ${file.name} (${state.image.width}x${state.image.height}). This image is already downloaded; use Skip to next pending to continue.`);
  } else {
    setStatus(`Loaded ${file.name} (${state.image.width}x${state.image.height}). Base tissue is ${currentLabelName(state.baseLabel)}.`);
  }
}

async function loadReviewIndex(index) {
  if (index < 0 || index >= state.reviewItems.length) return;
  const item = state.reviewItems[index];
  state.reviewIndex = index;
  state.currentIndex = state.batchFiles.indexOf(item.imageFile);
  state.imageName = item.imageFile.name;
  state.imageId = item.imageFile.name.replace(/\.[^.]+$/, "");
  state.currentMetadata = item.metadata;
  state.imageBitmap = await loadDrawableImage(item.imageFile);
  state.image = state.imageBitmap;
  state.tissuePolygons = [];
  state.currentPolygon = [];
  state.tissueLocked = true;
  state.viewZoom = 1;
  state.reviewMask = item.idMask ? await loadMaskFromFile(item.idMask, state.image.width, state.image.height) : null;
  state.tissueMask = state.reviewMask;
  els.imageId.value = state.imageId;
  renderSelectionMetadata();
  resizeCanvas(state.image.width, state.image.height, 96);
  resizePreviewCanvas(state.image.width, state.image.height);
  updateTissueLockUI();
  updateImageSelectorOptions();
  els.imageSelector.value = String(index);
  drawMainCanvas();
  const maskStatus = item.idMask ? `Mask: ${item.idMask.name}` : "No matching *_mask.png found";
  setStatus(`Reviewing ${item.imageFile.name}. ${maskStatus}.`);
}

function refreshImageSelector() {
  els.imageSelector.replaceChildren();
  if (state.mode === "review") {
    updateImageSelectorOptions();
    return;
  }
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
  if (state.mode === "review") {
    updateReviewSelectorOptions();
    return;
  }
  if (!state.batchFiles.length) return;
  els.imageSelector.replaceChildren();
  const pending = [];
  const inMemory = [];
  const downloaded = [];
  state.batchFiles.forEach((file, index) => {
    if (state.completedMasks.has(file.name)) {
      inMemory.push({ file, index });
    } else if (state.downloadedNames.has(file.name)) {
      downloaded.push({ file, index });
    } else {
      pending.push({ file, index });
    }
  });
  const groups = [
    ["Pending", pending],
    ["Confirmed (not downloaded)", inMemory],
    ["Downloaded", downloaded]
  ];
  for (const [label, items] of groups) {
    if (!items.length) continue;
    const group = document.createElement("optgroup");
    group.label = `${label} (${items.length})`;
    for (const { file, index } of items) {
      const option = document.createElement("option");
      option.value = String(index);
      const metadata = getSelectionMetadataForFile(state.batchManifest, file);
      option.textContent = metadata?.organZh ? `${file.name} - ${metadata.organZh}` : file.name;
      group.appendChild(option);
    }
    els.imageSelector.appendChild(group);
  }
  if (state.currentIndex >= 0) {
    els.imageSelector.value = String(state.currentIndex);
  }
}

function updateReviewSelectorOptions() {
  els.imageSelector.replaceChildren();
  if (!state.reviewItems.length) {
    const option = document.createElement("option");
    option.value = "-1";
    option.textContent = "No review loaded";
    els.imageSelector.appendChild(option);
    els.imageSelector.disabled = true;
    return;
  }
  els.imageSelector.disabled = false;
  const groups = [
    ["Matched masks", state.reviewItems.filter((item) => item.idMask)],
    ["Missing masks", state.reviewShowMissing ? state.reviewItems.filter((item) => !item.idMask) : []]
  ];
  for (const [label, items] of groups) {
    if (!items.length) continue;
    const group = document.createElement("optgroup");
    group.label = `${label} (${items.length})`;
    for (const item of items) {
      const index = state.reviewItems.indexOf(item);
      const option = document.createElement("option");
      option.value = String(index);
      const metadata = item.metadata;
      const suffix = item.idMask ? "" : " - missing mask";
      option.textContent = metadata?.organZh ? `${item.imageFile.name} - ${metadata.organZh}${suffix}` : `${item.imageFile.name}${suffix}`;
      group.appendChild(option);
    }
    els.imageSelector.appendChild(group);
  }
  if (state.reviewIndex >= 0) {
    els.imageSelector.value = String(state.reviewIndex);
  }
}

function renderSelectionMetadata() {
  if (!els.selectionMetadata) return;
  const metadata = state.currentMetadata;
  els.selectionMetadata.hidden = !metadata;
  els.selectionOrgan.textContent = metadata?.organZh || "";
  els.selectionCaption.textContent = metadata?.captionZh || "";
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

async function confirmTissue() {
  if (!state.imageBitmap || state.tissueLocked || state.downloadBlocked) return;
  completeCurrentPolygon();
  state.tissueLocked = true;
  state.completedMasks.set(state.imageName, {
    mask: cloneMask(state.tissueMask),
    width: state.image.width,
    height: state.image.height
  });
  state.masksSinceDownload += 1;
  updateProgressBadge();
  updateTissueLockUI();
  setStatus(`Tissue annotation confirmed for ${state.imageName}.`);

  if (state.masksSinceDownload >= AUTO_DOWNLOAD_THRESHOLD) {
    const downloaded = await autoDownloadMasks();
    if (!downloaded) return;
  }
  if (state.downloadBlocked) return;
  if (state.currentIndex + 1 < state.batchFiles.length) {
    loadBatchIndex(state.currentIndex + 1);
  } else {
    setStatus(`Tissue annotation confirmed for ${state.imageName}. Batch complete.`);
  }
}

function updateProgressBadge() {
  if (state.mode === "review") {
    updateModeUI();
    return;
  }
  const done = state.downloadedNames.size + state.completedMasks.size;
  const total = state.batchFiles.length;
  if (els.progressBadge) {
    els.progressBadge.textContent = `${done}/${total} done`;
  }
  if (els.skipToPending) {
    els.skipToPending.disabled = !state.batchFiles.length || done >= total;
  }
}

async function autoDownloadMasks() {
  if (state.completedMasks.size === 0) return true;
  state.downloadBlocked = true;
  if (els.downloadBanner) els.downloadBanner.hidden = false;
  setStatus(`Download required: preparing ${state.completedMasks.size} masks. Please allow the browser download to continue.`);
  try {
    await downloadZip();
    markCompletedMasksDownloaded();
    setStatus(`Downloaded and saved progress. ${state.downloadedNames.size} images completed so far.`);
    return true;
  } catch (error) {
    setStatus(`Download failed: ${formatErrorMessage(error)}. Click "Download masks" manually before continuing.`);
    return false;
  } finally {
    state.downloadBlocked = false;
    if (els.downloadBanner) els.downloadBanner.hidden = true;
  }
}

async function handleManualDownload() {
  if (state.completedMasks.size === 0) return;
  state.downloadBlocked = true;
  try {
    await downloadZip();
    markCompletedMasksDownloaded();
    setStatus(`Downloaded and saved progress. ${state.downloadedNames.size} images completed so far.`);
    skipToNextPending();
  } catch (error) {
    setStatus(`Download failed: ${formatErrorMessage(error)}.`);
  } finally {
    state.downloadBlocked = false;
  }
}

function markCompletedMasksDownloaded() {
  for (const name of state.completedMasks.keys()) {
    state.downloadedNames.add(name);
  }
  state.completedMasks.clear();
  state.masksSinceDownload = 0;
  saveProgress();
  updateProgressBadge();
  updateImageSelectorOptions();
  updateTissueLockUI();
}

function saveProgress() {
  try {
    const data = {
      batchId: state.batchId,
      downloaded: [...state.downloadedNames]
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (error) {
    // localStorage may be full or disabled; progress just won't persist.
  }
}

function loadProgress() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (data?.batchId !== state.batchId) {
      state.downloadedNames = new Set();
      return;
    }
    if (data?.downloaded && Array.isArray(data.downloaded)) {
      state.downloadedNames = new Set(data.downloaded);
    }
  } catch (error) {
    state.downloadedNames = new Set();
  }
}

function computeBatchId(files) {
  const names = files.map((f) => f.name).sort();
  return String(names.length) + ":" + (names.slice(0, 3).join(",") || "") + ":" + (names.slice(-3).join(",") || "");
}

function skipToNextPending() {
  if (state.mode === "review") {
    skipToNextReviewIssue();
    return;
  }
  if (!state.batchFiles.length) return;
  for (let i = 0; i < state.batchFiles.length; i += 1) {
    const name = state.batchFiles[i].name;
    if (!state.completedMasks.has(name) && !state.downloadedNames.has(name)) {
      loadBatchIndex(i);
      setStatus(`Skipped to next pending image: ${name}.`);
      return;
    }
  }
  setStatus("All images are completed. Download any remaining masks to finish.");
}

function skipToNextReviewIssue() {
  if (!state.reviewItems.length) return;
  const start = Math.max(state.reviewIndex + 1, 0);
  const nextMissing = state.reviewItems.findIndex((item, index) => index >= start && !item.idMask);
  if (nextMissing >= 0) {
    loadReviewIndex(nextMissing);
    setStatus(`Skipped to missing mask: ${state.reviewItems[nextMissing].imageFile.name}.`);
    return;
  }
  setStatus("No later images are missing masks.");
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
  if (state.mode === "review" && mode === "tissue") return;
  if (mode === "tissue" && state.tissueLocked) return;
  els.tissueMode.classList.toggle("active", mode === "tissue");
  drawMainCanvas();
}

function updateTissueLockUI() {
  updateModeUI();
  els.tissueSection.classList.toggle("locked", state.tissueLocked);
  els.clearTissue.disabled = state.tissueLocked;
  els.confirmTissue.disabled = state.tissueLocked;
  els.tissueMode.disabled = state.tissueLocked;
  renderLabelButtons();
  updateImageSelectorOptions();
  els.downloadZip.disabled = state.mode === "review" || state.completedMasks.size === 0;
}

function updateModeUI() {
  const isReview = state.mode === "review";
  if (els.reviewSection) els.reviewSection.hidden = !isReview;
  if (els.tissueSection) els.tissueSection.hidden = isReview;
  if (els.baseLabel) els.baseLabel.disabled = isReview;
  if (els.imageId) els.imageId.disabled = isReview;
  if (els.downloadZip) els.downloadZip.disabled = isReview || state.completedMasks.size === 0;
  if (els.skipToPending) els.skipToPending.textContent = isReview ? "Next missing mask" : "Skip to next pending";
  if (els.progressBadge && isReview) {
    const summary = summarizeReviewItems(state.reviewItems);
    els.progressBadge.textContent = `${summary.matched}/${summary.total} masks`;
  }
  renderReviewLegend();
}

function renderReviewSummary() {
  if (!els.reviewSummary) return;
  const summary = summarizeReviewItems(state.reviewItems);
  els.reviewSummary.hidden = state.mode !== "review";
  els.reviewSummary.textContent = `Review: ${summary.matched}/${summary.total} masks matched, ${summary.missing} missing.`;
}

function renderReviewLegend() {
  if (!els.reviewLabels) return;
  els.reviewLabels.replaceChildren();
  for (const [name, value, color] of tissueLabels) {
    const item = document.createElement("div");
    item.className = "legendItem";
    const swatch = document.createElement("span");
    swatch.className = "legendSwatch";
    swatch.style.background = color;
    const text = document.createElement("span");
    text.textContent = `${name} (${value})`;
    item.append(swatch, text);
    els.reviewLabels.appendChild(item);
  }
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
  const opacity = state.mode === "review" ? state.reviewOpacity : 0.3;
  if (opacity <= 0) return;
  const imageData = ctx.getImageData(offset.x, offset.y, width, height);
  for (let i = 0; i < state.tissueMask.length; i += 1) {
    const value = state.tissueMask[i];
    if (value === 0) continue;
    const [r, g, b] = hexToRgb(colorForTissue(value));
    const idx = i * 4;
    imageData.data[idx] = Math.round(imageData.data[idx] * (1 - opacity) + r * opacity);
    imageData.data[idx + 1] = Math.round(imageData.data[idx + 1] * (1 - opacity) + g * opacity);
    imageData.data[idx + 2] = Math.round(imageData.data[idx + 2] * (1 - opacity) + b * opacity);
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
  const files = [];
  for (const file of state.batchFiles) {
    const entry = state.completedMasks.get(file.name);
    if (!entry) continue;
    const imageId = sanitize(file.name.replace(/\.[^.]+$/, ""));
    const idMaskCanvas = maskToCanvas(entry.mask, entry.width, entry.height);
    const rgbMaskCanvas = maskToRgbCanvas(entry.mask, entry.width, entry.height);
    files.push({
      name: `masks/${imageId}_mask.png`,
      data: await canvasToUint8Array(idMaskCanvas)
    });
    files.push({
      name: `masks/${imageId}_mask_rgb.png`,
      data: await canvasToUint8Array(rgbMaskCanvas)
    });
  }
  if (!files.length) {
    setStatus("No confirmed masks to download yet.");
    return;
  }
  const zip = await buildZip(files);
  const link = document.createElement("a");
  link.href = URL.createObjectURL(zip);
  const batchNum = Math.floor(state.downloadedNames.size / AUTO_DOWNLOAD_THRESHOLD) + 1;
  link.download = `tissue_masks_batch_${batchNum}.zip`;
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

function maskToRgbCanvas(mask, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const imageData = new ImageData(width, height);
  for (let i = 0; i < mask.length; i += 1) {
    const [r, g, b] = hexToRgb(colorForTissue(mask[i]));
    const idx = i * 4;
    imageData.data[idx] = r;
    imageData.data[idx + 1] = g;
    imageData.data[idx + 2] = b;
    imageData.data[idx + 3] = 255;
  }
  canvas.getContext("2d").putImageData(imageData, 0, 0);
  return canvas;
}

async function loadMaskFromFile(file, expectedWidth, expectedHeight) {
  const image = await loadDrawableImage(file);
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const maskCtx = canvas.getContext("2d", { willReadFrequently: true });
  maskCtx.drawImage(image, 0, 0);
  const imageData = maskCtx.getImageData(0, 0, image.width, image.height);
  const source = imageData.data;
  const mask = new Uint8Array(expectedWidth * expectedHeight);
  const copyWidth = Math.min(expectedWidth, image.width);
  const copyHeight = Math.min(expectedHeight, image.height);
  for (let y = 0; y < copyHeight; y += 1) {
    for (let x = 0; x < copyWidth; x += 1) {
      const sourceIndex = (y * image.width + x) * 4;
      mask[y * expectedWidth + x] = source[sourceIndex];
    }
  }
  if (image.width !== expectedWidth || image.height !== expectedHeight) {
    setStatus(`Mask size ${image.width}x${image.height} does not match image ${expectedWidth}x${expectedHeight}; showing overlapping area only.`);
  }
  return mask;
}

async function canvasToUint8Array(canvas) {
  const dataUrl = canvas.toDataURL("image/png");
  const base64 = dataUrl.split(",")[1] || "";
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function canvasToBlob(canvas) {
  return new Promise((resolve) => {
    let resolved = false;
    const done = (blob) => { if (!resolved) { resolved = true; resolve(blob); } };
    const timer = setTimeout(() => {
      // fallback: encode via toDataURL if toBlob never fires
      try {
        const dataUrl = canvas.toDataURL("image/png");
        const base64 = dataUrl.split(",")[1];
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
        done(new Blob([bytes], { type: "image/png" }));
      } catch (e) {
        done(new Blob([], { type: "image/png" }));
      }
    }, 3000);
    try {
      canvas.toBlob((blob) => {
        clearTimeout(timer);
        done(blob);
      }, "image/png");
    } catch (e) {
      clearTimeout(timer);
      done(null);
    }
  });
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

function isImageFile(file) {
  return String(file.type || "").startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp)$/i.test(file.name || "");
}

function isCsvFile(file) {
  return /\.csv$/i.test(file.name || "") || file.type === "text/csv";
}

function clearFileInput(event) {
  event.currentTarget.value = "";
}

async function readFileAsText(file) {
  if (typeof file.text === "function") return file.text();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Failed to read CSV file."));
    reader.readAsText(file);
  });
}

async function loadDrawableImage(file) {
  if (typeof createImageBitmap === "function") {
    try {
      return await createImageBitmap(file);
    } catch {
      // Fall back to an HTMLImageElement so the UI can show a useful status instead of silently stopping.
    }
  }
  return loadImageElement(file);
}

function loadImageElement(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`The browser could not decode ${file.name}.`));
    };
    image.src = url;
  });
}

function formatErrorMessage(error) {
  return error?.message || String(error || "Unknown error");
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
