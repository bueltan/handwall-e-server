import asyncio
from typing import Dict, Optional

from modules.bridget_logger import BridgeLogger
from models.udp_config import UdpConfig

class JitterBuffer:
    """
    Stores incoming UDP audio packets and emits them in sequence order.

    Missing packets are replaced with silence after a configurable number
    of failed attempts to wait for the expected sequence number.
    """

    def __init__(self, config: UdpConfig, logger: BridgeLogger) -> None:
        self.config = config
        self.logger = logger

        self.buffer: Dict[int, bytes] = {}
        self.expected_seq: Optional[int] = None
        self.last_packet_size: int = config.packet_audio_size
        self.total_received: int = 0
        self.total_lost: int = 0
        self.missing_counter: int = 0
        self.input_closed: bool = False
        self.turn_index: int = 0

    def reset_for_cancel(self) -> None:
        """Clear all buffered state after a cancel event."""
        self.input_closed = True
        self.buffer.clear()
        self.expected_seq = None
        self.missing_counter = 0

    def open_new_turn_if_needed(self, seq: int) -> None:
        """
        Start a new input turn if the previous one was already closed.

        The new turn begins from the current packet sequence.
        """
        if self.input_closed:
            self.turn_index += 1
            self.buffer.clear()
            self.expected_seq = seq
            self.missing_counter = 0
            self.input_closed = False
            self.logger.log(
                f"New audio turn #{self.turn_index} -> initial seq={seq}",
                "TURN",
            )

    def register_packet(self, seq: int, audio: bytes) -> None:
        """Store an incoming packet in the jitter buffer."""
        self.last_packet_size = len(audio)
        self.total_received += 1

        if self.expected_seq is None:
            self.expected_seq = seq
            self.logger.log(f"First packet received -> seq={seq}", "INIT")

        self.buffer[seq] = audio

    def process_ordered_audio(self, audio_queue: asyncio.Queue) -> None:
        """
        Push in-order audio packets into the async audio queue.

        If the expected packet does not arrive after several retries,
        silence is inserted to preserve timing continuity.
        """
        if self.input_closed:
            return

        processed_any = False
        processed_count = 0

        while self.expected_seq in self.buffer and processed_count < 50:
            pcm = self.buffer.pop(self.expected_seq)
            audio_queue.put_nowait(pcm)
            self.expected_seq += 1
            self.missing_counter = 0
            processed_count += 1
            processed_any = True

        if not processed_any and self.expected_seq is not None and len(self.buffer) > 0:
            self.missing_counter += 1

            if self.missing_counter >= self.config.max_missing_before_loss:
                silence = b"\x00" * self.last_packet_size
                self.logger.log(
                    f"[LOSS] seq {self.expected_seq} lost -> filled with silence",
                    "LOSS",
                )
                audio_queue.put_nowait(silence)
                self.expected_seq += 1
                self.missing_counter = 0
                self.total_lost += 1

        if len(self.buffer) > self.config.max_buffer and self.expected_seq is not None:
            for old_seq in list(self.buffer.keys()):
                if old_seq < self.expected_seq - 10:
                    del self.buffer[old_seq]

    def close_input_turn(self, audio_queue: asyncio.Queue) -> None:
        """
        Flush any remaining in-order packets and mark the current turn as closed.

        Out-of-order packets still left in the buffer are discarded.
        """
        if self.expected_seq is None:
            self.input_closed = True
            self.missing_counter = 0
            self.buffer.clear()
            self.logger.log("Input turn closed with no pending audio", "TURN")
            return

        flushed = 0
        while self.expected_seq in self.buffer:
            pcm = self.buffer.pop(self.expected_seq)
            audio_queue.put_nowait(pcm)
            self.expected_seq += 1
            flushed += 1

        dropped = len(self.buffer)
        self.buffer.clear()

        self.input_closed = True
        self.expected_seq = None
        self.missing_counter = 0

        self.logger.log(
            f"Input turn closed | flushed={flushed} | dropped_out_of_order={dropped}",
            "TURN",
        )

    def loss_rate(self) -> float:
        """Return the packet loss rate percentage."""
        if self.total_received == 0:
            return 0.0
        return (self.total_lost / self.total_received) * 100