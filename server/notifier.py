"""Push notification system for fall alerts.

Supports:
  - Webhook (generic HTTP POST with JSON payload)
  - LINE Notify (LINE_NOTIFY_TOKEN)
  - Telegram Bot (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)

Configure via environment variables or Settings.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

MAX_RETRIES = 1
RETRY_DELAY_SEC = 1.0


@dataclass
class FallNotification:
    timestamp: float
    confidence: float
    head_height: float
    velocity: float
    alert_id: int = 0


class Notifier:
    """Sends fall alert notifications to configured channels."""

    def __init__(
        self,
        webhook_url: str = "",
        line_token: str = "",
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
    ):
        self.webhook_url = webhook_url
        self.line_token = line_token
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self._client = httpx.Client(timeout=10)
        self._channels = []
        if webhook_url:
            self._channels.append("webhook")
        if line_token:
            self._channels.append("line")
        if telegram_bot_token and telegram_chat_id:
            self._channels.append("telegram")
        if self._channels:
            logger.info("Notifier enabled: %s", ", ".join(self._channels))
        else:
            logger.info("Notifier: no channels configured (alerts stored in DB only)")

    @property
    def enabled(self) -> bool:
        return len(self._channels) > 0

    def send_fall_alert(self, notif: FallNotification) -> list[str]:
        """Send fall alert to all configured channels. Returns list of results."""
        results = []
        msg = (
            f"[WiFi Body] Fall detected!\n"
            f"Confidence: {notif.confidence:.0%}\n"
            f"Head height: {notif.head_height:.2f}m\n"
            f"Velocity: {notif.velocity:.2f}m/s"
        )
        payload = {
            "event": "fall_detected",
            "timestamp": notif.timestamp,
            "confidence": notif.confidence,
            "head_height": notif.head_height,
            "velocity": notif.velocity,
            "alert_id": notif.alert_id,
        }

        if "webhook" in self._channels:
            results.append(self._send_webhook(payload))
        if "line" in self._channels:
            results.append(self._send_line(msg))
        if "telegram" in self._channels:
            results.append(self._send_telegram(msg))
        return results

    def _send_with_retry(self, name: str, send_fn) -> str:
        """Execute *send_fn* with up to MAX_RETRIES on transient errors."""
        last_err = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                r = send_fn()
                r.raise_for_status()
                return f"{name}:ok ({r.status_code})"
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    logger.warning("%s transient error, retrying: %s", name, e)
                    time.sleep(RETRY_DELAY_SEC)
            except Exception as e:
                logger.error("%s failed: %s", name, e)
                return f"{name}:error ({e})"
        logger.error("%s failed after %d retries: %s", name, MAX_RETRIES, last_err)
        return f"{name}:error ({last_err})"

    def _send_webhook(self, payload: dict) -> str:
        return self._send_with_retry(
            "webhook",
            lambda: self._client.post(self.webhook_url, json=payload),
        )

    def _send_line(self, message: str) -> str:
        return self._send_with_retry(
            "line",
            lambda: self._client.post(
                "https://notify-api.line.me/api/notify",
                headers={"Authorization": f"Bearer {self.line_token}"},
                data={"message": message},
            ),
        )

    def _send_telegram(self, message: str) -> str:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        return self._send_with_retry(
            "telegram",
            lambda: self._client.post(
                url,
                json={"chat_id": self.telegram_chat_id, "text": message},
            ),
        )

    def close(self):
        self._client.close()
