import logging
import sys
import json
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record):

        log_record = {
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "time": self.formatTime(record),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_logging(log_file: str = "logs/app.log"):

    # Ensure logs directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = JsonFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs if re-initialized
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)