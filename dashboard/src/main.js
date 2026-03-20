// dashboard/src/main.js
/**
 * WiFi Body Dashboard — entry point.
 * Initializes EventBus, WebSocket client, demo data generator, and tab router.
 */
import { bus } from './events.js';
import { init as initDemoData } from './simulation/demo-data.js';

// Start demo data generator (auto-stops when WebSocket connects)
initDemoData();

console.log('WiFi Body Dashboard initialized');
console.log('EventBus ready:', bus);
