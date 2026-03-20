// dashboard/src/utils/format.js
/**
 * Number formatting utilities.
 */
export function formatBpm(value) {
  return value > 0 ? value.toFixed(1) : '--';
}

export function formatPercent(value) {
  return value > 0 ? (value * 100).toFixed(0) + '%' : '--';
}

export function formatConfidence(value) {
  if (value >= 0.8) return 'High';
  if (value >= 0.5) return 'Medium';
  return 'Low';
}
