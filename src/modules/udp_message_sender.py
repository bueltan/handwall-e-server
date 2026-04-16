import asyncio
import json
import socket
import struct
from typing import Optional, Tuple

from modules.bridget_logger import BridgeLogger


class UdpMessageSender:
    """Send JSON messages and paced raw audio packets to the last known UDP peer."""

    # 320 bytes = 160 mono PCM16 samples = 10 ms @ 16 kHz
    AUDIO_CHUNK_SIZE = 320
    OUTPUT_SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2

    # Exact realtime pacing.
    PACING_FACTOR = 1.0

    # Yield to the event loop every N packets to avoid bursty scheduling.
    YIELD_EVERY_CHUNKS = 20

    def __init__(self, logger: BridgeLogger) -> None:
        self.logger = logger
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        except Exception:
            pass

        self._paced_playback_id: Optional[int] = None
        self._next_send_time: Optional[float] = None

    @property
    def audio_packet_seconds(self) -> float:
        base_seconds = (
            self.AUDIO_CHUNK_SIZE
            / self.BYTES_PER_SAMPLE
            / self.OUTPUT_SAMPLE_RATE
        )
        return base_seconds * self.PACING_FACTOR

    def reset_audio_pacing(self, playback_id: int) -> None:
        self._paced_playback_id = playback_id
        self._next_send_time = None

    def send_json(self, payload: dict, addr: Optional[Tuple[str, int]]) -> None:
        if not addr:
            self.logger.log("Cannot send UDP JSON: remote address is unknown", "WARN")
            return

        try:
            data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.sock.sendto(data, addr)
            self.logger.log(
                f"UDP JSON sent to {addr}: {data.decode('utf-8', errors='ignore')}",
                "UDP",
            )
        except Exception as exc:
            self.logger.log(f"Failed to send UDP JSON: {exc}", "ERROR")

    async def send_audio_chunked(
        self,
        pcm_data: bytes,
        addr: Optional[Tuple[str, int]],
        playback_id: int,
        start_sequence: int,
        start_timestamp_ms: int,
    ) -> int:
        """
        Send assistant audio with header:
        [playback_id:u32][sequence:u32][timestamp_ms:u32][pcm...]

        Returns the next sequence number to continue from.
        """
        if not addr:
            self.logger.log("Cannot send UDP audio: remote address is unknown", "WARN")
            return start_sequence

        try:
            loop = asyncio.get_running_loop()

            if self._paced_playback_id != playback_id:
                self.reset_audio_pacing(playback_id)

            sequence = start_sequence
            sent_chunks = 0
            packet_ms = int(self.audio_packet_seconds * 1000)

            for offset in range(0, len(pcm_data), self.AUDIO_CHUNK_SIZE):
                chunk = pcm_data[offset: offset + self.AUDIO_CHUNK_SIZE]
                if not chunk:
                    continue

                chunk_timestamp_ms = start_timestamp_ms + (
                    (sequence - start_sequence) * packet_ms
                )

                header = struct.pack(
                    "<III",
                    playback_id,
                    sequence,
                    chunk_timestamp_ms,
                )

                packet = header + chunk

                now = loop.time()
                if self._next_send_time is None:
                    self._next_send_time = now

                delay = self._next_send_time - now
                if delay > 0:
                    await asyncio.sleep(delay)

                self.sock.sendto(packet, addr)

                # Advance pacing without accumulating a giant backlog.
                self._next_send_time = max(
                    self._next_send_time + self.audio_packet_seconds,
                    loop.time(),
                )

                sequence += 1
                sent_chunks += 1

                # Let the event loop breathe periodically.
                if sent_chunks % self.YIELD_EVERY_CHUNKS == 0:
                    await asyncio.sleep(0)

            self.logger.log(
                (
                    f"UDP audio paced send: {sent_chunks} chunks | "
                    f"chunk_bytes={self.AUDIO_CHUNK_SIZE} | "
                    f"packet_ms={self.audio_packet_seconds * 1000:.2f} | "
                    f"playback_id={playback_id}"
                ),
                "UDP",
            )

            return sequence

        except Exception as exc:
            self.logger.log(f"Failed to send UDP audio chunk: {exc}", "ERROR")
            return start_sequence

    def send_audio_end(self, addr: Optional[Tuple[str, int]]) -> None:
        if not addr:
            self.logger.log(
                "Cannot send UDP audio end marker: remote address is unknown",
                "WARN",
            )
            return

        try:
            self.sock.sendto(b"__END__", addr)
            self.logger.log(f"UDP audio end marker sent to {addr}", "UDP")
        except Exception as exc:
            self.logger.log(f"Failed to send UDP audio end marker: {exc}", "ERROR")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass