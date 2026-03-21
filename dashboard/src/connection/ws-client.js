// dashboard/src/connection/ws-client.js
/**
 * WebSocket client with v1 protocol, auto-reconnect, and heartbeat.
 */
import { bus } from '../events.js';
import { parseEnvelope } from './protocol.js';

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;

export class WsClient {
  constructor(url) {
    this._url = url;
    this._ws = null;
    this._reconnectAttempt = 0;
    this._capabilities = ['pose', 'vitals', 'csi', 'status', 'persons'];
  }

  connect() {
    try {
      this._ws = new WebSocket(this._url);
      this._ws.onopen = () => this._onOpen();
      this._ws.onmessage = (e) => this._onMessage(e.data);
      this._ws.onclose = () => this._onClose();
      this._ws.onerror = () => {}; // onclose will fire
    } catch (e) {
      this._scheduleReconnect();
    }
  }

  send(obj) {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(obj));
    }
  }

  _onOpen() {
    this._reconnectAttempt = 0;
    bus.emit('ws:connected');
    // Send v1 hello
    this.send({ v: 1, type: 'hello', capabilities: this._capabilities });
  }

  _onMessage(raw) {
    const parsed = parseEnvelope(raw);
    if (!parsed) return;

    if (parsed.version === 1) {
      if (parsed.type === 'ping') {
        this.send({ v: 1, type: 'pong', ts: Date.now() });
        return;
      }
      if (parsed.type === 'welcome') {
        bus.emit('ws:welcome', parsed.data);
        return;
      }
      // Emit by type: pose, vitals, csi, status, error
      bus.emit(parsed.type, parsed.data);
    } else {
      // v0 legacy: emit as individual streams
      const d = parsed.data;
      if (d.joints) bus.emit('pose', { joints: d.joints, confidence: 0 });
      if (d.vitals) bus.emit('vitals', d.vitals);
      if (d.csi_amplitudes) bus.emit('csi', { amplitudes: d.csi_amplitudes });
    }
  }

  _onClose() {
    bus.emit('ws:disconnected');
    this._scheduleReconnect();
  }

  _scheduleReconnect() {
    const delay = Math.min(
      RECONNECT_BASE_MS * Math.pow(2, this._reconnectAttempt),
      RECONNECT_MAX_MS,
    );
    this._reconnectAttempt++;
    setTimeout(() => this.connect(), delay);
  }

  // Heartbeat: server sends ping, client responds with pong in _onMessage
}
