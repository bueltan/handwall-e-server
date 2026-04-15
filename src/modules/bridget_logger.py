from pathlib import Path
from datetime import datetime


class BridgeLogger:
    """Simple logger that writes messages both to stdout and to a log file."""

    def __init__(self, log_file: str) -> None:
        base_dir = Path(__file__).resolve().parent.parent   # sube de src/ al root del proyecto
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        self.log_path = logs_dir / log_file

    def log(self, message: str, level: str = "INFO") -> None:
        """Write a formatted log line to stdout and to the log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] [{level}] {message}"
        print(line)

        try:
            with open(self.log_path, "a", encoding="utf-8") as file:
                file.write(line + "\n")
        except Exception as exc:
            print(f"Error writing log file: {exc}")