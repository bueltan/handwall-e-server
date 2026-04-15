import asyncio
import json
from typing import Optional, Tuple

from modules.bridget_logger import BridgeLogger
from modules.udp_server import UdpServer


class UdpMessageSender:
    """Send JSON messages and raw audio packets using the same UDP socket as UdpServer."""

    AUDIO_CHUNK_SIZE = 640
    OUTPUT_SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2  # PCM16 mono
    PACING_FACTOR = 1.10  # un poco más lento que realtime para remoto

    def __init__(self, logger: BridgeLogger, udp_server: UdpServer) -> None:
        self.logger = logger
        self.udp_server = udp_server

    @property
    def audio_packet_seconds(self) -> float:
        base_seconds = (
            self.AUDIO_CHUNK_SIZE
            / self.BYTES_PER_SAMPLE
            / self.OUTPUT_SAMPLE_RATE
        )
        return base_seconds * self.PACING_FACTOR

    def _sendto(self, payload: bytes, addr: Optional[Tuple[str, int]]) -> bool:
        if not addr:
            self.logger.log("Cannot send UDP packet: remote address is unknown", "WARN")
            return False

        if not self.udp_server.sock:
            self.logger.log("Cannot send UDP packet: UDP server socket is not ready", "WARN")
            return False

        try:
            self.udp_server.sock.sendto(payload, addr)
            return True
        except Exception as exc:
            self.logger.log(f"Failed to send UDP packet to {addr}: {exc}", "ERROR")
            return False

    def send_json(self, payload: dict, addr: Optional[Tuple[str, int]]) -> None:
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except Exception as exc:
            self.logger.log(f"Failed to encode UDP JSON: {exc}", "ERROR")
            return

        ok = self._sendto(data, addr)
        if ok:
            self.logger.log(
                f"UDP JSON sent to {addr}: {data.decode('utf-8', errors='ignore')}",
                "UDP",
            )

    async def send_audio_chunked(
        self,
        pcm_data: bytes,
        addr: Optional[Tuple[str, int]],
    ) -> None:
        if not addr:
            self.logger.log("Cannot send UDP audio: remote address is unknown", "WARN")
            return

        if not self.udp_server.sock:
            self.logger.log("Cannot send UDP audio: UDP server socket is not ready", "WARN")
            return

        try:
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            sent_chunks = 0

            for offset in range(0, len(pcm_data), self.AUDIO_CHUNK_SIZE):
                chunk = pcm_data[offset: offset + self.AUDIO_CHUNK_SIZE]
                if not chunk:
                    continue

                ok = self._sendto(chunk, addr)
                if not ok:
                    break

                sent_chunks += 1

                if sent_chunks <= 5 or sent_chunks % 20 == 0:
                    self.logger.log(
                        f"Sending audio chunk #{sent_chunks} to {addr} | bytes={len(chunk)}",
                        "UDP",
                    )

                target_time = start_time + (sent_chunks * self.audio_packet_seconds)
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
        ok = self._sendto(b"__END__", addr)
        if ok:
            self.logger.log(f"UDP audio end marker sent to {addr}", "UDP")

    def close(self) -> None:
        # El socket pertenece a UdpServer
        pass