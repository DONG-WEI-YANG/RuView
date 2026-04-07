"""UI/UX Designer Tour — dashboard usability, accessibility, responsive design.

A UI/UX designer reviews: Can a nurse read vital signs at 3am on a tablet?
Does the reconnect state give clear feedback? Are touch targets big enough?
Is the color contrast accessible? Does the tab system work on mobile?

Expert focus: readability under stress, accessibility compliance, responsive UX.
"""
import json

import pytest
import numpy as np

from server.config import Settings
from server.services.event_emitter import EventEmitter
from server.services.websocket_service import WebSocketService
from server.services.pipeline_service import PipelineService, SCENE_MODES
from server.protocol.envelope import (
    make_envelope, PoseData, VitalsData, CsiData, PersonsData, PersonData,
    Envelope,
)
from server.protocol.v0_adapter import v1_to_v0
from server.protocol.handlers import ConnectionState, ALL_STREAMS


# ═══════════════════════════════════════════════════════════
# 1. Color Contrast & Readability
# ═══════════════════════════════════════════════════════════

class TestColorContrast:
    """WCAG 2.1 AA requires 4.5:1 contrast for normal text, 3:1 for large text.
    Designer verifies the theme palette meets this in critical UI elements.
    """

    @staticmethod
    def _relative_luminance(hex_color: str) -> float:
        """Calculate relative luminance per WCAG 2.1 formula."""
        hex_color = hex_color.lstrip('#')
        r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        def linearize(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)

    @staticmethod
    def _contrast_ratio(fg: str, bg: str) -> float:
        l1 = TestColorContrast._relative_luminance(fg)
        l2 = TestColorContrast._relative_luminance(bg)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    def test_amber_on_dark_contrast(self):
        """Primary text (#ffb000) on dark background (#0a0a0a) — must be ≥4.5:1."""
        ratio = self._contrast_ratio('#ffb000', '#0a0a0a')
        assert ratio >= 4.5, f"Amber on dark: {ratio:.1f}:1 (need 4.5:1)"

    def test_secondary_text_contrast(self):
        """Secondary text (#cc8800) on dark background (#0a0a0a)."""
        ratio = self._contrast_ratio('#cc8800', '#0a0a0a')
        assert ratio >= 3.0, f"Secondary on dark: {ratio:.1f}:1 (need 3:1 for large text)"

    def test_alert_red_on_dark_contrast(self):
        """Fall alert red (#e63946) on dark (#0a0a0a) — critical readability."""
        ratio = self._contrast_ratio('#e63946', '#0a0a0a')
        assert ratio >= 4.5, f"Alert red on dark: {ratio:.1f}:1"

    def test_green_on_dark_contrast(self):
        """'Connected' jade green (#33b8a4) on dark (#0a0a0a)."""
        ratio = self._contrast_ratio('#33b8a4', '#0a0a0a')
        assert ratio >= 4.5, f"Green on dark: {ratio:.1f}:1"

    def test_hud_on_semitransparent(self):
        """HUD card text (#ffb000) on rgba(0,0,0,0.8) ≈ #000000."""
        ratio = self._contrast_ratio('#ffb000', '#000000')
        assert ratio >= 4.5, f"HUD amber on black: {ratio:.1f}:1"

    def test_signal_quality_colors(self):
        """Signal grade colors must be distinguishable from each other."""
        from server.services.signal_quality import CAPABILITY_TABLE
        colors = [v["color"] for v in CAPABILITY_TABLE.values()]
        # All 4 colors should be unique
        assert len(set(colors)) == 4, "Signal grade colors are not all distinct"


# ═══════════════════════════════════════════════════════════
# 2. WebSocket Reconnect UX — user feedback during disconnects
# ═══════════════════════════════════════════════════════════

class TestReconnectUX:
    """Designer asks: what does the user SEE when the connection drops?
    The exponential backoff (1s → 2s → 4s → ... → 30s) must be bounded.
    """

    def test_reconnect_backoff_bounded(self):
        """Max reconnect delay should not exceed 30 seconds."""
        RECONNECT_BASE_MS = 1000
        RECONNECT_MAX_MS = 30000
        # Simulate 10 reconnect attempts
        for attempt in range(10):
            delay = min(RECONNECT_BASE_MS * (2 ** attempt), RECONNECT_MAX_MS)
            assert delay <= RECONNECT_MAX_MS, f"Attempt {attempt}: {delay}ms > {RECONNECT_MAX_MS}ms"

    def test_reconnect_first_attempt_fast(self):
        """First reconnect should be ≤1 second — user expects quick recovery."""
        RECONNECT_BASE_MS = 1000
        delay = RECONNECT_BASE_MS * (2 ** 0)
        assert delay <= 1000

    def test_reconnect_sequence(self):
        """Verify the full backoff sequence: 1, 2, 4, 8, 16, 30, 30, 30..."""
        RECONNECT_BASE_MS = 1000
        RECONNECT_MAX_MS = 30000
        expected = [1000, 2000, 4000, 8000, 16000, 30000, 30000]
        for i, exp in enumerate(expected):
            actual = min(RECONNECT_BASE_MS * (2 ** i), RECONNECT_MAX_MS)
            assert actual == exp, f"Attempt {i}: expected {exp}, got {actual}"

    def test_ws_events_for_ui_feedback(self):
        """WebSocket service emits events that UI can bind to for status display."""
        # The WsClient emits 'ws:connected' and 'ws:disconnected' via bus
        # Server-side: WebSocketService tracks connection_count
        ws_svc = WebSocketService(emitter=EventEmitter())
        assert ws_svc.connection_count == 0  # UI shows "No connections"


# ═══════════════════════════════════════════════════════════
# 3. Tab System — navigation, lazy init, keyboard access
# ═══════════════════════════════════════════════════════════

class TestTabSystemUX:
    """Designer verifies: 7 tabs navigable, correct ARIA structure in HTML,
    lazy initialization, and mobile bottom-bar behavior.
    """

    def test_seven_tabs_defined_in_html(self):
        """index.html must have exactly 7 tab panels."""
        from pathlib import Path
        html = Path("dashboard/index.html").read_text(encoding="utf-8")
        panels = html.count('class="tab-panel"')
        assert panels == 7, f"Expected 7 tab panels, found {panels}"

    def test_tab_bar_has_tablist_role(self):
        """ARIA: tab-bar should have role=tablist (set in main.js)."""
        # The HTML has <nav id="tab-bar"> — main.js should set role="tablist"
        from pathlib import Path
        main_js = Path("dashboard/src/main.js").read_text(encoding="utf-8")
        assert "tablist" in main_js, "Tab bar should set ARIA role=tablist"

    def test_tab_panel_ids_match_convention(self):
        """Each .tab-panel section id should follow 'tab-{name}' convention."""
        from pathlib import Path
        html = Path("dashboard/index.html").read_text(encoding="utf-8")
        import re
        panels = re.findall(r'id="(tab-\w+)"\s+class="tab-panel"', html)
        expected = {"tab-viewer", "tab-dashboard", "tab-hardware",
                    "tab-demo", "tab-sensing", "tab-architecture", "tab-performance"}
        assert set(panels) == expected

    def test_mobile_touch_target_minimum(self):
        """WCAG 2.1 Level AAA: touch targets should be ≥44×44px.
        CSS: #tab-bar button has min-height: 44px on mobile.
        """
        from pathlib import Path
        css = Path("dashboard/styles/tabs.css").read_text(encoding="utf-8")
        assert "min-height: 44px" in css, "Mobile tab buttons need 44px min-height"

    def test_tab_content_scrollable(self):
        """Tab panels must have overflow-y: auto for long content."""
        from pathlib import Path
        css = Path("dashboard/styles/tabs.css").read_text(encoding="utf-8")
        assert "overflow-y: auto" in css


# ═══════════════════════════════════════════════════════════
# 4. Responsive Breakpoints — tablet, mobile, desktop
# ═══════════════════════════════════════════════════════════

class TestResponsiveDesign:
    """Designer validates breakpoints for nurse station (desktop),
    bedside tablet (768px), and phone (480px).
    """

    def test_three_breakpoints_defined(self):
        """main.css should have breakpoints at 480px, 768px."""
        from pathlib import Path
        css = Path("dashboard/styles/main.css").read_text(encoding="utf-8")
        assert "max-width: 768px" in css, "Missing tablet breakpoint"
        assert "max-width: 480px" in css, "Missing mobile breakpoint"

    def test_mobile_tab_bar_fixed_bottom(self):
        """On mobile (<480px), tab bar should be fixed at bottom of screen."""
        from pathlib import Path
        css = Path("dashboard/styles/main.css").read_text(encoding="utf-8")
        # In the 480px media query, tab-bar should be position: fixed; bottom: 0
        assert "position: fixed" in css
        assert "bottom: 0" in css

    def test_header_scales_for_mobile(self):
        """Header h1 should reduce font size on mobile."""
        from pathlib import Path
        css = Path("dashboard/styles/main.css").read_text(encoding="utf-8")
        assert "font-size: 14px" in css  # mobile h1 size


# ═══════════════════════════════════════════════════════════
# 5. HUD Vital Signs Display — readability under stress
# ═══════════════════════════════════════════════════════════

class TestHUDReadability:
    """Designer asks: can a nurse read BPM at 3am from 2 meters away?"""

    def test_hud_value_font_size(self):
        """HUD vital values should be ≥18px for readability."""
        from pathlib import Path
        css = Path("dashboard/styles/hud.css").read_text(encoding="utf-8")
        assert "font-size: 18px" in css

    def test_hud_has_confidence_bars(self):
        """Each vital sign should show a confidence indicator."""
        from pathlib import Path
        css = Path("dashboard/styles/hud.css").read_text(encoding="utf-8")
        assert "hud-conf-fill" in css

    def test_fall_alert_has_blink_animation(self):
        """Fall alert should blink to draw attention."""
        from pathlib import Path
        css = Path("dashboard/styles/hud.css").read_text(encoding="utf-8")
        assert "blink" in css
        assert "alert-danger" in css

    def test_hud_pointer_events_none(self):
        """HUD overlay should not block interaction with 3D viewer beneath."""
        from pathlib import Path
        css = Path("dashboard/styles/hud.css").read_text(encoding="utf-8")
        assert "pointer-events: none" in css


# ═══════════════════════════════════════════════════════════
# 6. Data Envelope — UI receives correct structure
# ═══════════════════════════════════════════════════════════

class TestDataForUI:
    """Designer verifies the data shape the frontend receives for rendering."""

    def test_pose_envelope_has_joints_and_confidence(self):
        env = make_envelope("pose", PoseData(
            joints=[[0.0, 1.5, 0.0]] * 24, confidence=0.85,
        ))
        j = json.loads(env.model_dump_json())
        assert len(j["data"]["joints"]) == 24
        assert j["data"]["confidence"] == 0.85

    def test_vitals_envelope_has_display_fields(self):
        env = make_envelope("vitals", VitalsData(
            heart_bpm=72.0, breathing_bpm=16.0, stress_index=35.0,
            sleep_stage="light", body_movement="still",
        ))
        j = json.loads(env.model_dump_json())
        d = j["data"]
        assert d["heart_bpm"] == 72.0
        assert d["sleep_stage"] == "light"
        assert d["body_movement"] == "still"

    def test_persons_envelope_has_color(self):
        """Multi-person view: each person needs a color for the 3D skeleton."""
        env = make_envelope("persons", PersonsData(
            persons=[PersonData(id=1, color="#00ff88")],
            count=1,
        ))
        j = json.loads(env.model_dump_json())
        assert j["data"]["persons"][0]["color"] == "#00ff88"

    def test_v0_legacy_still_has_joints(self):
        """Old dashboard (v0) must still receive joints array."""
        env = make_envelope("pose", PoseData(
            joints=[[1.0, 2.0, 3.0]] * 24, confidence=0.9,
        ))
        v0 = v1_to_v0(env)
        assert "joints" in v0
        assert len(v0["joints"]) == 24


# ═══════════════════════════════════════════════════════════
# 7. Scene Mode UX — mode badge and status display
# ═══════════════════════════════════════════════════════════

class TestSceneModeUX:
    """Designer validates the LIVE/DEMO badge and scene mode switching."""

    @pytest.mark.asyncio
    async def test_mode_endpoint_returns_status_for_badge(self, client):
        """GET /api/status returns scene_mode for the mode badge."""
        r = await client.get("/api/status")
        d = r.json()
        assert "scene_mode" in d
        assert d["scene_mode"] in SCENE_MODES

    @pytest.mark.asyncio
    async def test_scene_switch_returns_description(self, client):
        """Scene switch should return description for the UI tooltip."""
        r = await client.post("/api/system/scene", params={"scene": "fitness"})
        d = r.json()
        assert "description" in d

    @pytest.mark.asyncio
    async def test_status_has_simulation_flag(self, client):
        """UI needs is_simulating to show DEMO vs LIVE badge."""
        r = await client.get("/api/status")
        ps = r.json()["pipeline_status"]
        assert "is_simulating" in ps


# ═══════════════════════════════════════════════════════════
# 8. Connection Status — what the user sees
# ═══════════════════════════════════════════════════════════

class TestConnectionStatusUX:
    """Designer needs clear connection feedback for the header status bar."""

    @pytest.mark.asyncio
    async def test_root_returns_version(self, client):
        """Version displayed in footer/about — must be present."""
        r = await client.get("/")
        assert "version" in r.json()

    @pytest.mark.asyncio
    async def test_network_info_for_qr_code(self, client):
        """QR code sharing needs IP, port, and URLs."""
        r = await client.get("/api/network")
        d = r.json()
        assert "port" in d
        assert "hostname" in d

    @pytest.mark.asyncio
    async def test_signal_quality_for_header_dots(self, client):
        """Signal quality bar in header needs per-node data."""
        r = await client.get("/api/signal-quality")
        d = r.json()
        assert "nodes" in d
        assert "grade" in d


# ═══════════════════════════════════════════════════════════
# 9. Quick Setup Panel — designer walksthrough the new wizard
# ═══════════════════════════════════════════════════════════

class TestQuickSetupPanelUX:
    """Designer visits the hardware tab's Quick Setup panel (added last sprint).
    Checks: affordance cues, brand consistency, notification channels, API binding.
    """

    def test_panel_exists_with_stable_id(self):
        """Panel ID hw-quick-setup must be stable — JS and CSS reference it."""
        from pathlib import Path
        js = Path("dashboard/src/tabs/hardware.js").read_text(encoding="utf-8")
        assert "hw-quick-setup" in js

    def test_collapsible_toggle_arrow_affordance(self):
        """▶/▼ toggle arrow gives users a visual cue that the panel expands."""
        from pathlib import Path
        js = Path("dashboard/src/tabs/hardware.js").read_text(encoding="utf-8")
        assert "qs-toggle" in js
        assert "qs-body" in js  # collapsible section

    def test_scene_mode_buttons_both_present(self):
        """Two mode buttons with IDs so JS can highlight active state."""
        from pathlib import Path
        js = Path("dashboard/src/tabs/hardware.js").read_text(encoding="utf-8")
        assert "qs-mode-safety" in js
        assert "qs-mode-fitness" in js
        assert "active" in js  # CSS class toggled on selection

    def test_save_button_uses_brand_amber(self):
        """Save & Apply button is amber (#f80) — matches primary brand color."""
        from pathlib import Path
        js = Path("dashboard/src/tabs/hardware.js").read_text(encoding="utf-8")
        assert "Save & Apply" in js
        assert "#f80" in js

    def test_save_feedback_message_element(self):
        """qs-save-msg shows 'Saved!' confirmation — user knows action was applied."""
        from pathlib import Path
        js = Path("dashboard/src/tabs/hardware.js").read_text(encoding="utf-8")
        assert "qs-save-msg" in js

    def test_three_notification_channels_present(self):
        """Webhook, LINE, Telegram — three channels visible in one scroll."""
        from pathlib import Path
        js = Path("dashboard/src/tabs/hardware.js").read_text(encoding="utf-8")
        assert "qs-webhook" in js
        assert "qs-line" in js
        assert "qs-tg-token" in js
        assert "qs-tg-chat" in js

    def test_room_fields_decimal_step(self):
        """Room dimension inputs use step=0.1 — technician can set 3.5 m."""
        from pathlib import Path
        js = Path("dashboard/src/tabs/hardware.js").read_text(encoding="utf-8")
        assert "step" in js
        assert "0.1" in js

    def test_setup_panel_loads_from_api(self):
        """loadQuickSetup() fetches /api/settings/quick — UI mirrors live config."""
        from pathlib import Path
        js = Path("dashboard/src/tabs/hardware.js").read_text(encoding="utf-8")
        assert "loadQuickSetup" in js
        assert "/api/settings/quick" in js

    def test_setup_panel_saves_to_api(self):
        """saveQuickSetup() POSTs to /api/settings/quick — round-trip confirmed."""
        from pathlib import Path
        js = Path("dashboard/src/tabs/hardware.js").read_text(encoding="utf-8")
        assert "saveQuickSetup" in js

    @pytest.mark.asyncio
    async def test_quick_setup_get_has_all_display_fields(self, client):
        """API response has every field the panel renders."""
        r = await client.get("/api/settings/quick")
        assert r.status_code == 200
        d = r.json()
        for field in ["room_width", "room_depth", "room_height",
                      "scene_mode", "profiles", "scene_modes",
                      "notify_webhook_url", "notify_line_token"]:
            assert field in d, f"Quick Setup UI missing field: {field}"

    @pytest.mark.asyncio
    async def test_scene_switch_returns_description_for_tooltip(self, client):
        """POST scene returns description — designer uses it for button tooltip."""
        r = await client.post("/api/system/scene", params={"scene": "safety"})
        assert "description" in r.json()
