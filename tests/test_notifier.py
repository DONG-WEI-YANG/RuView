import time
import pytest
from server.notifier import Notifier, FallNotification


class TestNotifier:
    def test_no_channels_configured(self):
        n = Notifier()
        assert not n.enabled
        assert n._channels == []

    def test_webhook_channel(self):
        n = Notifier(webhook_url="http://example.com/hook")
        assert n.enabled
        assert "webhook" in n._channels

    def test_line_channel(self):
        n = Notifier(line_token="test-token")
        assert n.enabled
        assert "line" in n._channels

    def test_telegram_needs_both(self):
        # Bot token alone is not enough
        n1 = Notifier(telegram_bot_token="token")
        assert not n1.enabled
        # Both token + chat_id required
        n2 = Notifier(telegram_bot_token="token", telegram_chat_id="123")
        assert n2.enabled
        assert "telegram" in n2._channels

    def test_multiple_channels(self):
        n = Notifier(
            webhook_url="http://example.com",
            line_token="token",
        )
        assert len(n._channels) == 2

    def test_notification_payload(self):
        notif = FallNotification(
            timestamp=time.time(),
            confidence=0.95,
            head_height=0.3,
            velocity=1.5,
            alert_id=42,
        )
        assert notif.confidence == 0.95
        assert notif.alert_id == 42

    def test_send_no_channels_returns_empty(self):
        n = Notifier()
        results = n.send_fall_alert(FallNotification(
            timestamp=time.time(),
            confidence=0.9,
            head_height=0.3,
            velocity=1.0,
        ))
        assert results == []
