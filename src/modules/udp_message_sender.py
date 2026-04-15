import asyncio
import json
import socket
from typing import Optional, Tuple

from modules.bridget_logger import BridgeLogger


class UdpMessageSender:
    """Send JSON messages and raw audio packets to the last known UDP peer."""

    AUDIO_CHUNK_SIZE = 1200
    OUTPUT_SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2  # PCM16 mono
    PACING_FACTOR = 0.90  # < 1.0 = un poco más rápido que realtime

    def __init__(self, logger: BridgeLogger) -> None:
        self.logger = logger
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @property
    def audio_packet_seconds(self) -> float:
        base_seconds = (
            self.AUDIO_CHUNK_SIZE
            / self.BYTES_PER_SAMPLE
            / self.OUTPUT_SAMPLE_RATE
        )
        return base_seconds * self.PACING_FACTOR

    def send_json(self, payload: dict, addr: Optional[Tuple[str, int]]) -> None:
        if not addr:
            self.logger.log("Cannot send UDP JSON: remote address is unknown", "WARN")
            return

        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
    ) -> None:
        if not addr:
            self.logger.log("Cannot send UDP audio: remote address is unknown", "WARN")
            return

        try:
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            sent_chunks = 0

            for offset in range(0, len(pcm_data), self.AUDIO_CHUNK_SIZE):
                chunk = pcm_data[offset: offset + self.AUDIO_CHUNK_SIZE]
                if not chunk:
                    continue

                self.sock.sendto(chunk, addr)
                sent_chunks += 1

                target_time = start_time + (
                    sent_chunks * self.audio_packet_seconds
                )
                delay = target_time - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)

            self.logger.log(
                (
                    f"UDP audio paced send: {sent_chunks} chunks of "
                    f"{self.AUDIO_CHUNK_SIZE} bytes | "
                    f"packet_ms={self.audio_packet_seconds * 1000:.2f} | "
                    f"pacing_factor={self.PACING_FACTOR}"
                ),
                "UDP",
            )

        except Exception as exc:
            self.logger.log(f"Failed to send UDP audio chunk: {exc}", "ERROR")

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