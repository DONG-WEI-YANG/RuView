import asyncio
import struct
import numpy as np
import pytest
from server.csi_receiver import CSIReceiver
from server.csi_frame import MAGIC_HEADER
from server.config import Settings


def _build_fake_frame(node_id=1, seq=0):
    num_sub = 56
    header = struct.pack(
        "<IBIIQBBBB H",
        MAGIC_HEADER, 1, node_id, seq, 1000,
        0xD3, 0xD0, 6, 20, num_sub,
    )
    csi = b""
    for i in range(num_sub):
        csi += struct.pack("<hh", int(100 * np.cos(i)), int(100 * np.sin(i)))
    return header + csi


@pytest.mark.asyncio
async def test_receiver_processes_frames():
    settings = Settings(udp_port=15005)
    receiver = CSIReceiver(settings)
    received = []

    def on_frame(frame):
        received.append(frame)

    receiver.on_frame = on_frame
    task = asyncio.create_task(receiver.start())

    await asyncio.sleep(0.1)

    transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
        asyncio.DatagramProtocol,
        remote_addr=("127.0.0.1", 15005),
    )
    for i in range(3):
        transport.sendto(_build_fake_frame(node_id=1, seq=i))
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.2)
    receiver.stop()
    transport.close()

    assert len(received) == 3
    assert received[0].node_id == 1
    assert received[2].sequence == 2
