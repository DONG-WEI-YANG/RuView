// dashboard/src/utils/resize.js
/**
 * ResizeObserver wrapper — replaces per-frame resize hack.
 * Only fires callback when dimensions are positive (tab visible).
 */
export function observeResize(element, callback) {
  const ro = new ResizeObserver((entries) => {
    const { width, height } = entries[0].contentRect;
    if (width > 0 && height > 0) callback(width, height);
  });
  ro.observe(element);
  return () => ro.disconnect();
}
