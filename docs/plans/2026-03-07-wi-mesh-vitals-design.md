# Wi-Mesh-Inspired Vital Signs Enhancement

**Date**: 2026-03-07
**Status**: Implemented

## Context

Based on the Wi-Mesh article (Florida State / Rutgers, WiFi 6/7 AoA-based 3D body mesh at 2.4cm joint error), we extended the vital signs extraction pipeline with additional health metrics derivable from WiFi CSI signals.

## New Metrics

| Metric | Method | Output |
|---|---|---|
| HRV (RMSSD) | Inter-beat interval analysis from heart signal peaks | milliseconds |
| HRV (SDNN) | Standard deviation of NN intervals | milliseconds |
| Stress Index | Inverse mapping from RMSSD (low HRV = high stress) | 0-100 score |
| Motion Intensity | RMS of broadband (1-8 Hz) CSI variance | 0-100 score |
| Body Movement | Threshold classification from motion intensity | still/micro/gross |
| Breath Regularity | Coefficient of variation of breath-to-breath intervals | 0-1 |
| Sleep Stage | Heuristic from breath regularity + motion level | awake/light/deep/rem |

## Architecture

All metrics computed inside `VitalSignsExtractor.update()` using the existing CSI buffer and already-computed bandpassed signals. No new classes or data flows needed.

## Files Modified

- `server/vital_signs.py` — 5 new computation methods, extended `_result()` dict
- `dashboard/index.html` — Health Metrics panel with 4 cards (HRV, Stress, Motion, Sleep) + HRV/Stress HUD overlay on 3D viewer
- `dashboard/vitals-hud.js` — DEMO simulation of new metrics, `renderHealthMetrics()`, `setExtendedVitals()` API
- `dashboard/styles.css` — Health card styles, stress bar, mini-bar, HRV HUD styling
- `dashboard/skeleton3d.js` — Forward extended vitals through WebSocket
- `tests/test_vital_signs.py` — 5 new tests for HRV, stress, motion, sleep, regularity

## Test Results

113 tests passing (108 original + 5 new).
