# Architecture & UI Optimization Design

**Date:** 2026-03-20
**Status:** Approved
**Strategy:** Protocol-First (define contract → refactor server → refactor dashboard)

## Goals

- **Primary:** Improve maintainability through modularization and decoupling
- **Secondary:** Lay foundation for better UX via clean architecture
- **Constraint:** REST API endpoints remain unchanged; WebSocket v0/v1 coexist during migration

## Current State (Problems)

### Server
- `api.py` is 669 lines with 20+ REST endpoints, WebSocket handler, UDP receiver, simulation loop, storage, and notifications all in one file
- Global `_state` dict with 20+ keys acts as a mutable god-object with no type safety
- Fall detector created in both `create_app()` and `PosePipeline` (dual truth source)
- WebSocket broadcasts call `numpy.tolist()` + `json.dumps()` per frame with no batching
- No request validation (Pydantic models missing for POST bodies)
- Simulation loop tightly coupled to api.py as a background task

### Dashboard
- `index.html` is 2,006 lines (all 7 tabs inlined)
- `skeleton3d.js` is 719 lines (Three.js + WebSocket + hardware config + DOM manipulation)
- `vitals-hud.js` is 692 lines (simulation + rendering + WebSocket mixed)
- Global variables couple modules; no event bus
- Per-frame canvas resize hack with `requestAnimationFrame` delay for tab switching
- No accessibility (ARIA labels, keyboard navigation, colorblind support)

### Protocol
- No versioning or schema validation
- All data (joints + vitals + CSI) multiplexed into single JSON payload every frame
- No heartbeat; dead connections only cleaned up on send failure
- Client ignores all incoming text except broadcast payloads

## Design

### Phase 1: WebSocket Protocol v1

#### Envelope Format

All v1 messages share a common envelope:

```json
{
  "v": 1,
  "ts": 1710900000000,
  "seq": 12345,
  "type": "pose | vitals | csi | status | error",
  "data": { ... }
}
```

- `v` — Protocol version. Server detects client's hello to decide v0 or v1.
- `ts` — Server-side millisecond timestamp (single clock source).
- `seq` — Monotonically increasing sequence number. Client can detect dropped messages.
- `type` — Message type. Replaces current all-in-one payload approach.

#### Stream Separation

| Type | Frequency | Content |
|------|-----------|---------|
| `pose` | 20 Hz | joints coordinates + per-joint confidence |
| `vitals` | 1 Hz | heart BPM, breathing BPM, HRV, stress, motion, sleep stage |
| `csi` | 5 Hz | CSI amplitudes for waterfall visualization |
| `status` | 0.2 Hz | pipeline state, connection count, mode |
| `error` | on-demand | error message + error code |

#### Connection Handshake

```
Client → Server:  {"v": 1, "type": "hello", "capabilities": ["pose", "vitals", "csi"]}
Server → Client:  {"v": 1, "type": "welcome", "server_version": "0.2.0", "streams": ["pose", "vitals", "csi"]}
```

- Client declares which streams to subscribe to (bandwidth savings).
- **v0 detection:** If no `hello` message received within 5 seconds after connection, treat as v0 client and fallback to legacy format. The current v0 client never sends messages, so detection is by absence of hello, not by message content.
- v0 adapter on server side: thin transform merging separated pose+vitals+csi back into single payload.
- `seq` is a 64-bit integer, no wraparound expected (at 20 Hz, overflows in ~14 billion years).

#### Heartbeat

```
Server → Client:  {"v": 1, "type": "ping", "ts": ...}
Client → Server:  {"v": 1, "type": "pong", "ts": ...}
```

30 seconds without pong → server removes dead connection (replaces current send-fail-then-cleanup).

#### Implementation Details

- `server/protocol/envelope.py` — Pydantic models for all message types
- `server/protocol/v0_adapter.py` — Bidirectional v0 ↔ v1 conversion
- `server/protocol/handlers.py` — Handle hello, pong, and future client messages

### Phase 2: Server Service Layer

#### New Directory Structure

```
server/
├── api.py                  ← Slim: ~80 lines (FastAPI app + lifespan + router mounts)
├── services/
│   ├── __init__.py
│   ├── container.py        ← DI container, manages service lifecycles
│   ├── pipeline_service.py ← CSI receive → signal processing → inference
│   ├── websocket_service.py← Connection mgmt, v0/v1 dispatch, heartbeat
│   ├── vitals_service.py   ← Vital signs extraction + multi-person tracking
│   ├── calibration_service.py ← Background calibration workflow
│   ├── storage_service.py  ← SQLite persistence
│   └── notification_service.py ← Webhook/Line/Telegram push
├── routes/
│   ├── __init__.py
│   ├── ws.py               ← WebSocket endpoint (/ws/pose)
│   ├── calibration.py      ← /api/calibration/*
│   ├── system.py           ← /api/status, /api/system/*
│   └── data.py             ← /api/sessions, /api/export
├── protocol/               ← (from Phase 1)
├── signal_processor.py     ← Unchanged
├── vital_signs.py          ← Unchanged (wrapped by VitalsService)
├── pipeline.py             ← Unchanged (wrapped by PipelineService)
├── config.py               ← Unchanged
└── ...
```

#### DI Container

```python
@dataclass
class ServiceContainer:
    settings: Settings
    pipeline: PipelineService
    websocket: WebSocketService
    vitals: VitalsService
    calibration: CalibrationService
    storage: StorageService
    notifier: NotificationService

    async def startup(self): ...
    async def shutdown(self): ...
```

- FastAPI lifespan creates `ServiceContainer`, stores in `app.state`.
- Routes access services via `Depends(get_container)`.
- Services communicate through interfaces, never touching each other's internal state.

#### Service Responsibilities

| Service | Input | Output | Dependencies |
|---------|-------|--------|-------------|
| PipelineService | CSI frames (UDP) | Joint coordinates | SignalProcessor, PoseModel |
| WebSocketService | joints/vitals/csi | v0 or v1 messages | ProtocolEnvelope, V0Adapter |
| VitalsService | CSI amplitudes | Vital signs dict | VitalSignsExtractor |
| CalibrationService | Calibration commands | Calibration results | PipelineService |
| StorageService | pose/vitals events | SQLite writes | None |
| NotificationService | Alerts | webhook/Line/Telegram | None |

#### Event-Driven Data Flow

```
CSI UDP → PipelineService.on_frame()
    ├→ emit("pose", joints)     → WebSocketService → broadcast to clients
    ├→ emit("csi", amplitudes)  → WebSocketService → broadcast to subscribers
    └→ VitalsService.push(csi)
        └→ emit("vitals", data) → WebSocketService → broadcast at 1Hz
```

Lightweight async event emitter (asyncio-based pub/sub). No external message broker needed.

**Event emitter implementation:** A simple callback-based emitter in `server/services/event_emitter.py`. Uses `asyncio.create_task` to invoke subscribers. If a subscriber raises, log the error and continue (don't crash the emitter loop). No third-party dependency needed.

**`_on_csi_frame` migration:** The current 78-line orchestration function feeds CSI to calibration, collector, vitals, pipeline, storage, notifier, and broadcast. This logic distributes across services:
- Calibration feed → `CalibrationService.on_frame()`
- Collector feed → `PipelineService.on_frame()` (data collection is a pipeline concern)
- Vitals feed → `VitalsService.push()`
- Storage throttle → `StorageService` subscribes to pose/vitals events
- Notification → `NotificationService` subscribes to alert events
- Broadcast → `WebSocketService` subscribes to all stream events

**Phase 2 sequencing:** Migrate route-group by route-group to keep the 125-test suite green throughout: calibration routes first → data routes → system routes → WebSocket handler (largest, last).

**Closure hazard:** Current route handlers close over the `settings` parameter of `create_app()`. After migration, all routes use `Depends(get_container)` instead — no closures over function arguments.

#### Key Changes

1. `api.py` shrinks from 669 to ~80 lines (app creation, lifespan, router mounting only)
2. Fall detector lives only inside `PipelineService` (no more dual creation)
3. Simulation loop moves into `PipelineService` (not an api.py background task)
4. WebSocket broadcast becomes event-driven (PipelineService emits → WebSocketService subscribes)

### Phase 3: Dashboard Vite + ES6 Modules

#### New Directory Structure

```
dashboard/
├── index.html              ← Slim: ~50 lines (<div id="app"> + <script type="module">)
├── vite.config.js
├── package.json
├── src/
│   ├── main.js             ← Entry: init app, event bus, tab router
│   ├── events.js           ← Lightweight EventEmitter (app-wide communication backbone)
│   ├── connection/
│   │   ├── ws-client.js    ← WebSocket connection, v1 hello/pong, auto-reconnect
│   │   └── protocol.js     ← v1 envelope parsing, v0 fallback detection
│   ├── scene/
│   │   ├── three-setup.js  ← Three.js init, camera, lighting, controls
│   │   ├── skeleton.js     ← 24-joint skeleton rendering
│   │   ├── body-mesh.js    ← SMPL body mesh
│   │   └── room.js         ← Room geometry + node markers
│   ├── vitals/
│   │   ├── hud.js          ← HUD overlay rendering (pure rendering)
│   │   ├── waveform.js     ← Breathing/heartbeat waveform canvas
│   │   ├── waterfall.js    ← CSI waterfall spectrogram
│   │   └── heatmap.js      ← Presence heatmap
│   ├── tabs/
│   │   ├── tab-manager.js  ← Tab switching + lazy initialization
│   │   ├── viewer.js       ← 3D Viewer tab controller
│   │   ├── dashboard.js    ← Dashboard tab controller
│   │   ├── hardware.js     ← Hardware tab (profile selector, AP diagram)
│   │   ├── demo.js         ← Live Demo tab
│   │   ├── sensing.js      ← Sensing tab
│   │   ├── architecture.js ← Architecture tab
│   │   └── performance.js  ← Performance tab
│   ├── simulation/
│   │   └── demo-data.js    ← Client-side demo data generation (extracted from vitals-hud)
│   └── utils/
│       ├── resize.js       ← ResizeObserver wrapper (replaces per-frame resize hack)
│       └── format.js       ← Number formatting (BPM, %, etc.)
├── styles/
│   ├── main.css            ← CSS custom properties + global reset
│   ├── tabs.css            ← Tab navigation styles
│   ├── hud.css             ← HUD overlay
│   ├── cards.css           ← Health cards, status cards
│   └── effects.css         ← CRT scanline, glow, animations
└── public/
    └── models/             ← GLB/GLTF static assets
```

#### Core Patterns

**EventBus (replaces global variable coupling):**
```javascript
// events.js
class EventBus {
  on(event, handler) { ... }
  off(event, handler) { ... }
  emit(event, data) { ... }
}
export const bus = new EventBus();
```

All modules communicate through the bus. `ws-client.js` emits `pose`, `vitals`, `csi` events. Tab controllers subscribe to what they need.

**Tab Lazy Initialization (fixes canvas 0-dimension bug):**
```javascript
// tab-manager.js
const initialized = new Set();
function switchTab(tabId) {
  if (!initialized.has(tabId)) {
    tabs[tabId].init();       // First open: create canvas/Three.js
    initialized.add(tabId);
  }
  tabs[tabId].activate();     // ResizeObserver handles dimensions
}
```

No more `requestAnimationFrame` delay hack. Three.js scene only created when Viewer tab first opened.

**ResizeObserver (replaces per-frame resize):**
```javascript
// utils/resize.js
export function observeResize(element, callback) {
  const ro = new ResizeObserver(entries => {
    const { width, height } = entries[0].contentRect;
    if (width > 0 && height > 0) callback(width, height);
  });
  ro.observe(element);
  return () => ro.disconnect();
}
```

**WebSocket Client (v1 protocol + auto-reconnect):**
```javascript
// connection/ws-client.js
export class WsClient {
  connect(url) { ... }
  onOpen() {
    this.send({ v: 1, type: 'hello', capabilities: ['pose', 'vitals', 'csi'] });
  }
  onMessage(envelope) {
    if (!envelope.v) return this.handleV0(envelope);  // v0 fallback
    bus.emit(envelope.type, envelope.data);            // v1: emit by type
  }
  startHeartbeat() { ... }    // 30s ping/pong
  scheduleReconnect() { ... } // exponential backoff
}
```

#### Vite Configuration

```javascript
// vite.config.js
export default {
  root: 'dashboard',
  build: {
    outDir: '../dist/dashboard',
    rollupOptions: { input: 'dashboard/index.html' }
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true }
    }
  }
}
```

- Dev: `npx vite` with HMR, proxy to FastAPI backend (port 5173 → 8000)
- Prod: `npx vite build` → bundled to `dist/dashboard/`. FastAPI static mount changes from `dashboard/` to `dist/dashboard/` in production. `start.bat` updated to run `npx vite build` before starting server if `dist/` doesn't exist.

### Phase 4: Cleanup & Polish

1. Remove v0 adapter if no legacy clients remain
2. Delete old monolithic files (`skeleton3d.js`, `vitals-hud.js`, original `styles.css`)
3. Verify all 125 existing tests pass against new service layer
4. Add integration tests for v1 protocol handshake and stream separation

## Test Strategy Per Phase

| Phase | Test Approach |
|-------|-------------|
| Phase 1 | Unit tests for v0 adapter: feed v0 payloads → verify v1 output and vice versa. Test envelope serialization with Pydantic model validation. |
| Phase 2 | Existing 125 tests must pass without modification (they test HTTP endpoints, not internals). Add service-level unit tests for each new service. Migrate route-by-route, running full test suite after each route group. |
| Phase 3 | Dashboard currently has no JS test infrastructure. Vite enables adding `vitest` trivially. Scope: add unit tests for `ws-client.js` (protocol parsing) and `events.js` (EventBus). Visual/rendering tests are out of scope. |
| Phase 4 | Full regression: all server tests + new protocol tests + manual dashboard smoke test across all 7 tabs. |

## Migration Safety

Each phase produces a fully working system:

| Phase | Server | Dashboard | Compatibility |
|-------|--------|-----------|--------------|
| Phase 1 | Sends v0 + v1 | Unchanged (receives v0) | Full backward compat |
| Phase 2 | Service layer, sends v0 + v1 | Unchanged (receives v0) | Full backward compat |
| Phase 3 | Service layer, sends v0 + v1 | ES6 modules, receives v1 | Both versions work |
| Phase 4 | Service layer, v1 only | ES6 modules, v1 only | Clean, v0 removed |

## Files Changed

### New Files
- `server/services/container.py`
- `server/services/pipeline_service.py`
- `server/services/websocket_service.py`
- `server/services/vitals_service.py`
- `server/services/calibration_service.py`
- `server/services/storage_service.py`
- `server/services/notification_service.py`
- `server/routes/ws.py`
- `server/routes/calibration.py`
- `server/routes/system.py`
- `server/routes/data.py`
- `server/protocol/envelope.py`
- `server/protocol/v0_adapter.py`
- `server/protocol/handlers.py`
- `dashboard/vite.config.js`
- `dashboard/package.json`
- `dashboard/src/` (all modules listed above)
- `dashboard/styles/` (split CSS files)

### Modified Files
- `server/api.py` — Slim down from 669 to ~80 lines
- `server/__main__.py` — Update to use ServiceContainer
- `dashboard/index.html` — Slim down from 2,006 to ~50 lines

### Deleted Files (Phase 4)
- `dashboard/skeleton3d.js` — Replaced by `src/scene/*` + `src/connection/*`
- `dashboard/vitals-hud.js` — Replaced by `src/vitals/*` + `src/simulation/*`
- `dashboard/styles.css` — Replaced by `styles/*`

## Out of Scope

- Visual redesign or theme changes (retro-cyberpunk stays)
- New features (no new tabs, endpoints, or metrics)
- TypeScript migration (future consideration after Vite is in place)
- External message broker (asyncio event emitter is sufficient)
- REST API endpoint changes (all existing routes preserved)
