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
  if (!tabs[tabId] || tabId === activeTabId) return;

  // Deactivate old tab
  if (activeTabId && tabs[activeTabId]) {
    const oldPanel = document.getElementById('tab-' + activeTabId);
    if (oldPanel) oldPanel.classList.remove('active');
    tabs[activeTabId].deactivate();
  }

  // Init if first time
  if (!initialized.has(tabId)) {
    tabs[tabId].init();
    initialized.add(tabId);
  }

  // Activate new tab with fade-in
  const newPanel = document.getElementById('tab-' + tabId);
  if (newPanel) {
    newPanel.style.opacity = '0';
    newPanel.classList.add('active');
    // Trigger reflow then fade in
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { newPanel.style.opacity = '1'; });
    });
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
