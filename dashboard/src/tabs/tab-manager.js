// dashboard/src/tabs/tab-manager.js
/**
 * Tab switching with lazy initialization.
 * Each tab controller must export: { id, label, init(), activate(), deactivate() }
 */
import { bus } from '../events.js';

const tabs = {};
const initialized = new Set();
let activeTabId = null;

export function registerTab(controller) {
  tabs[controller.id] = controller;
}

export function switchTab(tabId) {
  if (!tabs[tabId]) return;
  if (activeTabId && tabs[activeTabId]) {
    tabs[activeTabId].deactivate();
  }
  if (!initialized.has(tabId)) {
    tabs[tabId].init();
    initialized.add(tabId);
  }
  tabs[tabId].activate();
  activeTabId = tabId;
  bus.emit('tab:changed', tabId);
}

export function getRegisteredTabs() {
  return Object.values(tabs);
}

export function getActiveTabId() {
  return activeTabId;
}
