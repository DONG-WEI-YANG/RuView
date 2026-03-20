// dashboard/src/events.js
/**
 * Lightweight EventEmitter — app-wide communication backbone.
 * Replaces global variable coupling between modules.
 */
export class EventBus {
  constructor() {
    this._handlers = {};
  }

  on(event, handler) {
    if (!this._handlers[event]) this._handlers[event] = [];
    this._handlers[event].push(handler);
  }

  off(event, handler) {
    const list = this._handlers[event];
    if (!list) return;
    const idx = list.indexOf(handler);
    if (idx >= 0) list.splice(idx, 1);
  }

  emit(event, data) {
    const list = this._handlers[event];
    if (!list) return;
    for (const handler of list) {
      try {
        handler(data);
      } catch (e) {
        console.error(`EventBus error on '${event}':`, e);
      }
    }
  }
}

export const bus = new EventBus();
