// dashboard/src/main.js
/**
 * WiFi Body Dashboard — entry point.
 * Registers tab controllers, builds the tab bar, connects WebSocket,
 * and starts the demo data generator.
 */
import { bus } from './events.js';
import { WsClient } from './connection/ws-client.js';
import { registerTab, switchTab, getRegisteredTabs } from './tabs/tab-manager.js';
import { init as startDemoData } from './simulation/demo-data.js';
import viewer from './tabs/viewer.js';
import dashboard from './tabs/dashboard.js';
import hardware from './tabs/hardware.js';
import demo from './tabs/demo.js';
import sensing from './tabs/sensing.js';
import architecture from './tabs/architecture.js';
import performance from './tabs/performance.js';

// Register all tabs
[viewer, dashboard, hardware, demo, sensing, architecture, performance]
  .forEach(registerTab);

// Build tab bar
const tabBar = document.getElementById('tab-bar');
tabBar.setAttribute('role', 'tablist');
getRegisteredTabs().forEach((tab) => {
  const btn = document.createElement('button');
  btn.textContent = tab.label;
  btn.dataset.tab = tab.id;
  btn.setAttribute('role', 'tab');
  btn.setAttribute('aria-selected', 'false');
  btn.id = 'tab-btn-' + tab.id;
  btn.setAttribute('aria-controls', 'tab-' + tab.id);
  btn.addEventListener('click', () => switchTab(tab.id));
  tabBar.appendChild(btn);
});

// Keyboard navigation for tab bar
tabBar.addEventListener('keydown', (e) => {
  const tabs = Array.from(tabBar.querySelectorAll('button'));
  const idx = tabs.indexOf(document.activeElement);
  if (idx < 0) return;
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
    e.preventDefault();
    const next = tabs[(idx + 1) % tabs.length];
    next.focus();
    switchTab(next.dataset.tab);
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
    e.preventDefault();
    const prev = tabs[(idx - 1 + tabs.length) % tabs.length];
    prev.focus();
    switchTab(prev.dataset.tab);
  }
});

// Highlight active tab and set ARIA attributes
bus.on('tab:changed', (tabId) => {
  tabBar.querySelectorAll('button').forEach((btn) => {
    const isActive = btn.dataset.tab === tabId;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });
  // Set tabpanel roles on each panel
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.setAttribute('role', 'tabpanel');
    p.setAttribute('aria-labelledby', 'tab-btn-' + p.id.replace('tab-', ''));
  });
});

// Connect WebSocket
const wsUrl = `ws://${location.hostname}:${location.port || 8000}/ws/pose`;
const client = new WsClient(wsUrl);
client.connect();

// Connection status indicator
const statusEl = document.getElementById('connection-status');
const modeEl = document.getElementById('mode-badge');

bus.on('ws:connected', () => {
  statusEl.textContent = 'Connected';
  statusEl.className = 'connected';
  // Check if server is in simulation mode → start demo data
  fetch('/api/status').then(r => r.json()).then(data => {
    if (data.pipeline_status && data.pipeline_status.is_simulating) {
      modeEl.textContent = 'DEMO';
      modeEl.className = 'demo';
      startDemoData();
    } else {
      modeEl.textContent = 'LIVE';
      modeEl.className = 'live';
    }
  }).catch(() => {});
});

bus.on('ws:disconnected', () => {
  statusEl.textContent = 'Waiting for server...';
  statusEl.className = 'disconnected';
  modeEl.textContent = '';
});

// Don't auto-start demo data — wait for server connection to decide

// Default to viewer tab
switchTab('viewer');
