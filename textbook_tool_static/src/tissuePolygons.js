export function buildMaskFromPolygons(width, height, polygons, background = 0) {
  const mask = new Uint8Array(width * height);
  if (background !== 0) {
    mask.fill(background);
  }
  for (const polygon of polygons) {
    rasterizePolygon(mask, width, height, polygon.points, polygon.label);
  }
  return mask;
}

export function removeLastPolygonForLabel(polygons, label) {
  const index = findLastIndex(polygons, (polygon) => polygon.label === label);
  if (index < 0) return polygons.slice();
  return [...polygons.slice(0, index), ...polygons.slice(index + 1)];
}

export function rasterizePolygon(mask, width, height, points, label) {
  if (!points || points.length < 3) return mask;
  const bounds = polygonBounds(points, width, height);
  for (let y = bounds.minY; y <= bounds.maxY; y += 1) {
    for (let x = bounds.minX; x <= bounds.maxX; x += 1) {
      if (pointInPolygon(x + 0.5, y + 0.5, points)) {
        mask[y * width + x] = label;
      }
    }
  }
  return mask;
}

export function pointInPolygon(x, y, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
    const xi = points[i].x;
    const yi = points[i].y;
    const xj = points[j].x;
    const yj = points[j].y;
    const intersects = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function polygonBounds(points, width, height) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    minX: clamp(Math.floor(Math.min(...xs)), 0, width - 1),
    maxX: clamp(Math.ceil(Math.max(...xs)), 0, width - 1),
    minY: clamp(Math.floor(Math.min(...ys)), 0, height - 1),
    maxY: clamp(Math.ceil(Math.max(...ys)), 0, height - 1)
  };
}

function findLastIndex(items, predicate) {
  for (let i = items.length - 1; i >= 0; i -= 1) {
    if (predicate(items[i], i)) return i;
  }
  return -1;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
