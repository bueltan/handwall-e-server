import wave
from pathlib import Path


class OutputWaveWriter:
    """Handles writing assistant output audio to a WAV file."""

    def __init__(self, wav_file: str, input_sample_rate: int) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        wav_dir = base_dir / "audios"
        wav_dir.mkdir(parents=True, exist_ok=True)

        self.wav_path = wav_dir / wav_file
        self._wave = wave.open(str(self.wav_path), "wb")
        self._wave.setnchannels(1)
        self._wave.setsampwidth(2)
        self._wave.setframerate(input_sample_rate)
        self._closed = False

    def write(self, pcm_data: bytes) -> None:
        """Append raw PCM16 mono audio to the WAV file."""
        if self._closed:
            raise RuntimeError("Cannot write to a closed WAV file.")
        self._wave.writeframes(pcm_data)

    def close(self) -> None:
        """Close the WAV file safely."""
        if not self._closed:
            self._wave.close()
            self._closed = True

    def __enter__(self) -> "OutputWaveWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()