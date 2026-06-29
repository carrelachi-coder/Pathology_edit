const ID_MASK_SUFFIX = "_mask";
const RGB_MASK_SUFFIX = "_mask_rgb";

export function buildReviewItems(imageFiles, maskFiles, manifest = emptyManifest()) {
  const idMasks = new Map();
  const rgbMasks = new Map();

  for (const file of maskFiles) {
    const parsed = parseMaskFileName(file?.name || "");
    if (!parsed) continue;
    const target = parsed.kind === "rgb" ? rgbMasks : idMasks;
    if (!target.has(parsed.baseKey)) target.set(parsed.baseKey, file);
  }

  return sortFiles(imageFiles).map((imageFile) => {
    const imageKeys = imageMaskKeys(imageFile);
    const idMask = firstMatchingFile(idMasks, imageKeys);
    const rgbMask = firstMatchingFile(rgbMasks, imageKeys);
    return {
      imageFile,
      idMask,
      rgbMask,
      metadata: getSelectionMetadataForFile(manifest, imageFile),
      status: idMask ? "matched" : "missing"
    };
  });
}

export function imageMaskKeys(file) {
  const base = withoutExtension(file?.name || "");
  return [
    normalizeFileKey(base),
    normalizeFileKey(sanitizeExportName(base))
  ].filter((key, index, keys) => key && keys.indexOf(key) === index);
}

function firstMatchingFile(filesByKey, keys) {
  for (const key of keys) {
    const file = filesByKey.get(key);
    if (file) return file;
  }
  return null;
}

export function sanitizeExportName(value) {
  return String(value).replace(/[^a-zA-Z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "textbook_image";
}

export function parseMaskFileName(fileName) {
  const base = withoutExtension(basename(fileName));
  const lower = base.toLowerCase();
  if (lower.endsWith(RGB_MASK_SUFFIX)) {
    return { baseKey: normalizeFileKey(base.slice(0, -RGB_MASK_SUFFIX.length)), kind: "rgb" };
  }
  if (lower.endsWith(ID_MASK_SUFFIX)) {
    return { baseKey: normalizeFileKey(base.slice(0, -ID_MASK_SUFFIX.length)), kind: "id" };
  }
  return null;
}

export function summarizeReviewItems(items) {
  const total = items.length;
  const matched = items.filter((item) => item.idMask).length;
  const missing = total - matched;
  return { total, matched, missing };
}

export function emptyManifest() {
  return { byFileKey: new Map(), rowCount: 0 };
}

export function parseSelectionManifest(text) {
  const rows = parseCsvRows(String(text || ""));
  if (rows.length === 0) return emptyManifest();

  const headers = rows[0].map((header) => normalizeHeader(header));
  const byFileKey = new Map();
  let rowCount = 0;

  for (const row of rows.slice(1)) {
    if (row.every((cell) => String(cell || "").trim() === "")) continue;
    rowCount += 1;

    const pngFile = valueFor(row, headers, "png_file");
    const baseName = valueFor(row, headers, "base_name");
    const metadata = {
      pngFile,
      baseName,
      organZh: valueFor(row, headers, "organ_zh"),
      captionZh: valueFor(row, headers, "caption_zh")
    };

    for (const key of manifestKeys(pngFile, baseName)) {
      if (!byFileKey.has(key)) byFileKey.set(key, metadata);
    }
  }

  return { byFileKey, rowCount };
}

export function getSelectionMetadataForFile(manifest, file) {
  if (!manifest?.byFileKey || !file) return null;
  const candidates = [
    file.name,
    file.webkitRelativePath,
    basename(file.webkitRelativePath || ""),
    withoutExtension(file.name || ""),
    withoutExtension(basename(file.webkitRelativePath || ""))
  ];

  for (const candidate of candidates) {
    const metadata = manifest.byFileKey.get(normalizeFileKey(candidate));
    if (metadata) return metadata;
  }
  return null;
}

export function parseCsvRows(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  let i = text.charCodeAt(0) === 0xfeff ? 1 : 0;

  for (; i < text.length; i += 1) {
    const char = text[i];
    if (inQuotes) {
      if (char === "\"") {
        if (text[i + 1] === "\"") {
          cell += "\"";
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        cell += char;
      }
      continue;
    }

    if (char === "\"") {
      inQuotes = true;
    } else if (char === ",") {
      row.push(cell);
      cell = "";
    } else if (char === "\n" || char === "\r") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
      if (char === "\r" && text[i + 1] === "\n") i += 1;
    } else {
      cell += char;
    }
  }

  if (cell !== "" || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }

  return rows;
}

export function manifestKeys(pngFile, baseName) {
  const values = [
    pngFile,
    basename(pngFile),
    withoutExtension(pngFile),
    withoutExtension(basename(pngFile)),
    baseName,
    baseName ? `${baseName}.png` : ""
  ];
  return [...new Set(values.map(normalizeFileKey).filter(Boolean))];
}

export function valueFor(row, headers, name) {
  const index = headers.indexOf(name);
  return index >= 0 ? String(row[index] || "").trim() : "";
}

export function normalizeHeader(value) {
  return String(value || "").replace(/^\ufeff/, "").trim().toLowerCase();
}

export function normalizeFileKey(value) {
  return withoutExtension(basename(String(value || "").trim())).toLowerCase();
}

export function basename(value) {
  return String(value || "").split(/[\\/]/).pop() || "";
}

export function withoutExtension(value) {
  return String(value || "").replace(/\.[^.\\/]+$/, "");
}

export function sortFiles(files) {
  return [...files].sort((a, b) => (a.name || "").localeCompare(b.name || "", undefined, { numeric: true, sensitivity: "base" }));
}
