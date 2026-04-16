import asyncio
import base64
from builtins import int
from datetime import datetime
import json
import os
import random

import websockets
from websockets.exceptions import ConnectionClosed

from models.udp_config import UdpConfig
from models.xai_config import XaiConfig
from modules.bridget_logger import BridgeLogger
from modules.output_waver_wrriter import OutputWaveWriter
from modules.udp_message_sender import UdpMessageSender
from modules.udp_server import UdpServer


class XaiRealtimeClient:
    """Manages a resilient realtime WebSocket connection to xAI."""

    def __init__(
        self,
        config: XaiConfig,
        udp_config: UdpConfig,
        logger: BridgeLogger,
        wav_writer: OutputWaveWriter,
        udp_message_sender: UdpMessageSender,
        udp_server: UdpServer,
    ) -> None:
        self.config = config
        self.udp_config = udp_config
        self.logger = logger
        self.wav_writer = wav_writer
        self.udp_message_sender = udp_message_sender
        self.udp_server = udp_server

        self.ws = None
        self.ws_send_lock = asyncio.Lock()
        self.connected_event = asyncio.Event()
        self.stop_event = asyncio.Event()

        self.current_assistant_text = ""
        self.pending_assistant_text = ""

        self.output_audio_bytes = 0
        self.output_audio_chunks = 0

        self.output_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.output_audio_done_event = asyncio.Event()
        self.output_audio_done_event.set()
        self.output_turn_finished = False

        self.reconnect_attempt = 0
        self.current_playback_id = 0
        self.output_sequence = 0
        self.output_started_at_ms = 0

    async def connect(self) -> None:
        await self.connect_once()

    def _reset_output_turn_state(self) -> None:
        self.current_assistant_text = ""
        self.pending_assistant_text = ""
        self.output_audio_bytes = 0
        self.output_audio_chunks = 0
        self.output_turn_finished = False
        self.output_audio_done_event.clear()

        self.current_playback_id += 1
        self.output_sequence = 0
        self.output_started_at_ms = 0

        self.udp_message_sender.reset_audio_pacing(self.current_playback_id)

    def _clear_output_queue(self) -> None:
        while True:
            try:
                self.output_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _fail_current_turn(self) -> None:
        self.current_assistant_text = ""
        self.pending_assistant_text = ""
        self.output_audio_bytes = 0
        self.output_audio_chunks = 0
        self.output_turn_finished = False

        self._clear_output_queue()

        if not self.output_audio_done_event.is_set():
            self.output_audio_done_event.set()

        try:
            if self.udp_server.remote_addr:
                self.udp_message_sender.send_json(
                    {
                        "type": "bridge_status",
                        "value": "xai_disconnected",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                    self.udp_server.remote_addr,
                )
                self.udp_message_sender.send_audio_end(
                    self.udp_server.remote_addr,
                )
        except Exception as exc:
            self.logger.log(
                f"Failed to notify UDP peer on disconnect: {exc}",
                "WARN",
            )

    async def _safe_send(self, payload: dict) -> None:
        if not self.connected_event.is_set() or self.ws is None:
            raise RuntimeError("WebSocket is not connected")

        try:
            if getattr(self.ws, "closed", False):
                raise RuntimeError("WebSocket is already closed")
        except Exception:
            pass

        async with self.ws_send_lock:
            await self.ws.send(json.dumps(payload))

    async def _send_session_update(self) -> None:
        await self._safe_send(
            {
                "type": "session.update",
                "session": {
                    "voice": self.config.voice,
                    "instructions": self.config.agent_prompt,
                    "turn_detection": None,
                    "tools": [
                        {"type": "web_search"},
                        {"type": "x_search"},
                    ],
                    "input_audio_transcription": {
                        "model": "grok-2-audio"
                    },
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.udp_config.input_sample_rate,
                            }
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.udp_config.output_sample_rate,
                            }
                        },
                    },
                },
            }
        )

    async def connect_once(self) -> None:
        api_key = os.environ.get("XAI_API_KEY", self.config.default_api_key)
        headers = {"Authorization": f"Bearer {api_key}"}

        self.ws = await websockets.connect(
            self.config.realtime_url,
            additional_headers=headers,
            open_timeout=20,
            close_timeout=10,
            ping_interval=15,
            ping_timeout=30,
            max_queue=64,
        )

        self.connected_event.set()
        await self._send_session_update()

        self.logger.log(
            (
                "Connected to xAI realtime | "
                f"input_rate={self.udp_config.input_sample_rate} | "
                f"output_rate={self.udp_config.output_sample_rate}"
            ),
            "START",
        )

        if self.reconnect_attempt > 0:
            self.logger.log(
                f"Reconnected successfully after {self.reconnect_attempt} failed attempt(s)",
                "XAI",
            )

        self.reconnect_attempt = 0

        if self.udp_server.remote_addr:
            self.udp_message_sender.send_json(
                {
                    "type": "bridge_status",
                    "value": "xai_connected",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
                self.udp_server.remote_addr,
            )

    async def wait_until_connected(self) -> None:
        await self.connected_event.wait()

    async def send_audio_append(self, pcm_data: bytes) -> bool:
        if not self.connected_event.is_set():
            self.logger.log(
                "Dropping input audio: websocket disconnected",
                "WARN",
            )
            return False

        try:
            await self._safe_send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm_data).decode("utf-8"),
                }
            )
            return True
        except Exception as exc:
            self.logger.log(f"send_audio_append failed: {exc}", "ERROR")
            return False

    async def send_commit(self) -> bool:
        if not self.connected_event.is_set():
            self.logger.log(
                "Cannot commit: websocket disconnected",
                "WARN",
            )
            return False

        self._reset_output_turn_state()

        try:
            await self._safe_send({"type": "input_audio_buffer.commit"})
            await self._safe_send({"type": "response.create"})
            self.logger.log("COMMIT sent to xAI", "TURN")
            return True
        except Exception as exc:
            self.logger.log(f"send_commit failed: {exc}", "ERROR")
            self._fail_current_turn()
            return False

    async def close(self) -> None:
        self.stop_event.set()
        self.connected_event.clear()

        self._fail_current_turn()

        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception as exc:
                self.logger.log(
                    f"Error while closing websocket: {exc}",
                    "WARN",
                )
            finally:
                self.ws = None

    async def _send_pending_assistant_text(self) -> None:
        assistant_text = self.pending_assistant_text.strip()
        if not assistant_text or not self.udp_server.remote_addr:
            return

        self.logger.log(f"Assistant response: {assistant_text}")
        self.logger.log(
            f"UDP target addr: {self.udp_server.remote_addr}",
            "UDP",
        )
        self.udp_message_sender.send_json(
            {
                "type": "assistant_response",
                "value": assistant_text,
                "final": True,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            self.udp_server.remote_addr,
        )
        self.pending_assistant_text = ""

    async def udp_audio_sender(self) -> None:
        while True:
            pcm_data = await self.output_audio_queue.get()

            if pcm_data is None:
                break

            try:
                self.output_sequence = await self.udp_message_sender.send_audio_chunked(
                    pcm_data,
                    self.udp_server.remote_addr,
                    playback_id=self.current_playback_id,
                    start_sequence=self.output_sequence,
                    start_timestamp_ms=self.output_started_at_ms,
                )
            except Exception as exc:
                self.logger.log(f"UDP audio sender error: {exc}", "ERROR")

            if self.output_audio_queue.empty() and self.output_turn_finished:
                try:
                    await asyncio.sleep(0.15)

                    self.udp_message_sender.send_audio_end(
                        self.udp_server.remote_addr,
                    )
                    await self._send_pending_assistant_text()
                except Exception as exc:
                    self.logger.log(f"Failed to finalize output: {exc}", "WARN")

                self.output_turn_finished = False
                self.output_audio_done_event.set()

    async def receive_events(self) -> None:
        try:
            async for raw_event in self.ws:
                event = json.loads(raw_event)
                event_type = event.get("type")

                if event_type == "session.created":
                    session_id = event.get("session", {}).get("id")
                    self.logger.log(f"Session created: {session_id}", "XAI")

                elif event_type == "session.updated":
                    self.logger.log("Session updated", "XAI")

                elif event_type == "input_audio_buffer.committed":
                    self.logger.log("xAI confirmed commit", "XAI")

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "").strip()
                    self.logger.log(f"User transcription: {transcript}", "USER")

                elif event_type == "response.output_audio.delta":
                    pcm_data = base64.b64decode(event["delta"])

                    self.output_audio_chunks += 1
                    self.output_audio_bytes += len(pcm_data)
                    #self.wav_writer.write(pcm_data)

                    if (
                        self.output_audio_chunks <= 5
                        or self.output_audio_chunks % 25 == 0
                    ):
                        samples = len(pcm_data) // 2
                        approx_ms = (
                            samples / self.udp_config.output_sample_rate
                        ) * 1000.0
                        self.logger.log(
                            (
                                f"Output audio chunk #{self.output_audio_chunks} | "
                                f"bytes={len(pcm_data)} | "
                                f"samples={samples} | "
                                f"~{approx_ms:.2f} ms"
                            ),
                            "AUDIO",
                        )

                    await self.output_audio_queue.put(pcm_data)

                elif event_type == "response.output_audio_transcript.delta":
                    delta_text = event.get("delta", "")
                    if delta_text:
                        self.current_assistant_text += delta_text
                        print(delta_text, end="", flush=True)

                elif event_type == "response.output_audio_transcript.done":
                    transcript = event.get("transcript", "").strip()
                    if transcript:
                        self.current_assistant_text = transcript
                        self.logger.log(
                            f"Assistant transcript done: {transcript}",
                            "ASSISTANT",
                        )

                elif event_type == "response.output_audio.done":
                    print()

                    total_samples = self.output_audio_bytes // 2
                    total_ms = (
                        total_samples / self.udp_config.output_sample_rate
                    ) * 1000.0

                    self.logger.log(
                        (
                            "Response audio completed | "
                            f"chunks={self.output_audio_chunks} | "
                            f"bytes={self.output_audio_bytes} | "
                            f"samples={total_samples} | "
                            f"~{total_ms:.2f} ms"
                        ),
                        "ASSISTANT",
                    )

                    self.output_turn_finished = True

                    if self.output_audio_queue.empty():
                        try:
                            await asyncio.sleep(0.15)

                            self.udp_message_sender.send_audio_end(
                                self.udp_server.remote_addr,
                            )
                            await self._send_pending_assistant_text()
                        except Exception as exc:
                            self.logger.log(
                                f"Failed to finalize output immediately: {exc}",
                                "WARN",
                            )

                        self.output_turn_finished = False
                        self.output_audio_done_event.set()

                elif event_type == "response.done":
                    usage = event.get("usage", {})
                    self.logger.log(
                        f"Response completed | tokens={usage.get('total_tokens')}",
                        "XAI",
                    )

                    assistant_text = self.current_assistant_text.strip()
                    self.pending_assistant_text = assistant_text

                    if self.output_audio_chunks == 0 and not assistant_text:
                        self.logger.log(
                            "Response completed with no audio chunks and no assistant transcript",
                            "WARN",
                        )
                        self.output_audio_done_event.set()

                    if self.output_audio_chunks == 0 and assistant_text:
                        await self._send_pending_assistant_text()

                elif event_type == "ping":
                    pass

                elif event_type == "error":
                    self.logger.log(
                        f"xAI error: {json.dumps(event, ensure_ascii=False)}",
                        "ERROR",
                    )

                else:
                    self.logger.log(
                        f"Unhandled xAI event: type={event_type} keys={list(event.keys())}",
                        "DEBUG",
                    )

        except ConnectionClosed as exc:
            self.logger.log(
                (
                    "websocket closed: "
                    f"code={getattr(exc, 'code', None)} "
                    f"reason={getattr(exc, 'reason', None)}"
                ),
                "ERROR",
            )
            raise

        except Exception as exc:
            self.logger.log(f"websocket receiver error: {exc}", "ERROR")
            raise

    async def run_forever(self) -> None:
        while not self.stop_event.is_set():
            try:
                await self.connect_once()
                await self.receive_events()

            except Exception as exc:
                self.connected_event.clear()
                self._fail_current_turn()

                if self.stop_event.is_set():
                    break

                self.logger.log(f"WebSocket disconnected: {exc}", "ERROR")

                self.reconnect_attempt += 1
                base_delay = min(30, 2 ** min(self.reconnect_attempt, 5))
                jitter = random.uniform(0, 1.0)
                delay = base_delay + jitter

                self.logger.log(
                    f"Reconnecting in {delay:.1f}s (attempt {self.reconnect_attempt})",
                    "WARN",
                )

                try:
                    if self.ws is not None:
                        await self.ws.close()
                except Exception:
                    pass
                finally:
                    self.ws = None

                await asyncio.sleep(delay)