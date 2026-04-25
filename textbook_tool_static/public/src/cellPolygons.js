export const TARGET_TUMOR_CELLS = 10;

export function polygonArea(points) {
  if (!points || points.length < 3) return 0;
  let sum = 0;
  for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
    sum += points[j].x * points[i].y - points[i].x * points[j].y;
  }
  return Math.abs(sum) / 2;
}

export function equivalentDiameter(points) {
  const area = polygonArea(points);
  if (area <= 0) return null;
  return 2 * Math.sqrt(area / Math.PI);
}

export function medianCellDiameter(cells) {
  const values = cells
    .map((cell) => equivalentDiameter(cell.points))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  if (!values.length) return null;
  const mid = Math.floor(values.length / 2);
  return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

export function remainingCellText(count, target = TARGET_TUMOR_CELLS) {
  return `Cells remaining: ${Math.max(0, target - count)}/${target}`;
}
