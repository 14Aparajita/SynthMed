# src/utils/logging.py - FIXED VERSION

import logging
import sys
from pathlib import Path

def setup_logging(log_dir: str = "outputs/logs", level: int = logging.INFO):
    """Configure logging for the project."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("synthmed")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler (UTF-8 encoding)
    fh = logging.FileHandler(
        Path(log_dir) / "experiment.log",
        encoding='utf-8'
    )
    fh.setLevel(level)
    
    # Console handler with error handling
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    
    # Set encoding for console handler
    if hasattr(ch, 'stream') and hasattr(ch.stream, 'reconfigure'):
        try:
            ch.stream.reconfigure(encoding='utf-8')
        except:
            pass
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add error handler for console encoding issues
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except UnicodeEncodeError:
                # Fall back to ASCII
                msg = self.format(record).encode('ascii', errors='replace').decode('ascii')
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger