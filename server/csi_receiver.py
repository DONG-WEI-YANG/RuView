"""Async UDP receiver for multi-node ESP32 CSI data."""
import asyncio
import logging
from collections import defaultdict
from typing import Callable

from server.config import Settings
from server.csi_frame import CSIFrame, parse_csi_frame

logger = logging.getLogger(__name__)


class CSIReceiver:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.on_frame: Callable[[CSIFrame], None] | None = None
        self._transport = None
        self._running = False
        self.node_stats: dict[int, int] = defaultdict(int)

    async def start(self):
        loop = asyncio.get_event_loop()
        self._running = True
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _CSIProtocol(self),
            local_addr=(self.settings.udp_host, self.settings.udp_port),
        )
        logger.info(
            "CSI receiver listening on %s:%d",
            self.settings.udp_host,
            self.settings.udp_port,
        )
        while self._running:
            await asyncio.sleep(0.1)

    def stop(self):
        self._running = False
        if self._transport:
            self._transport.close()

    def _handle_data(self, data: bytes):
        frame = parse_csi_frame(data)
        if frame is None:
            return
        self.node_stats[frame.node_id] += 1
        if self.on_frame:
            self.on_frame(frame)


class _CSIProtocol(asyncio.DatagramProtocol):
    def __init__(self, receiver: CSIReceiver):
        self.receiver = receiver

    def datagram_received(self, data: bytes, addr):
        self.receiver._handle_data(data)
