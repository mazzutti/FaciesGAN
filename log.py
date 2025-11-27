import sys
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Writable(Protocol):
    def write(self, *args, **kwargs) -> object: ...

    def flush(self) -> None: ...


class OutputLogger(object):
    def __init__(self, file: str | None = None, mode: str | None = None, buffer: str | None = "") -> None:
        """
        Initialize the OutputLogger.

        Attributes:
            file (Optional[str]): The log file name.
            mode (Optional[str]): The mode in which to open the log file.
            buffer (Optional[str]): The buffer to store log data before writing to the file.
        """
        self.file = file
        self.mode = mode
        self.buffer = buffer

    def set_log_file(self, filename: str, mode: str = "at") -> None:
        """
        Set the log file and write any buffered data to it.

        Args:
            filename (str): The name of the log file.
            mode (str): The mode in which to open the log file. Default is "at".
        """
        assert self.file is None
        self.file = filename
        self.mode = mode
        if self.buffer is not None:
            with open(self.file, self.mode) as f:
                f.write(self.buffer)
                self.buffer = None

    def write(self, data: str) -> None:
        """
        Write data to the log file and buffer.

        Args:
            data (str): The data to write.
        """
        # Do not save tqdm print
        if data.startswith("\r"):
            return
        if self.file is not None:
            with open(self.file, self.mode) as f:
                f.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self) -> None:
        """
        Flush the log file.
        """
        if self.file is not None:
            with open(self.file, self.mode) as f:
                f.flush()


class TeeOutputStream(object):
    def __init__(self, child_streams: Sequence[Writable], auto_flush: bool = False) -> None:
        """
        Initialize the TeeOutputStream.

        Args:
            child_streams (list): List of child streams to write to.
            auto_flush (bool): Whether to automatically flush after each write. Default is False.
        """
        self.child_streams = child_streams
        self.auto_flush = auto_flush
        self.buffer = ""

    def write(self, data: str) -> None:
        """
        Write data to all child streams.

        Args:
            data (str): The data to write.
        """
        if data is not None:
            for stream in self.child_streams:
                stream.write(data)
            if self.auto_flush:
                self.flush()

    def flush(self) -> None:
        """
        Flush all child streams.
        """
        for stream in self.child_streams:
            stream.flush()


def init_output_logging(filename: str, mode: str = "at", output_logger: OutputLogger | None = None) -> None:
    """
    Initialize output logging to a file.

    Args:
        filename (str): The name of the file to log output to.
        mode (str): The mode in which to open the log file. Default is "at".
        output_logger (OutputLogger, optional): An instance of OutputLogger. Default is None.
    """
    if output_logger is None:
        output_logger = OutputLogger()
        sys.stdout = TeeOutputStream([sys.stdout, output_logger], auto_flush=True)
        sys.stderr = TeeOutputStream([sys.stderr, output_logger], auto_flush=True)
    output_logger.set_log_file(filename, mode)


def format_time(seconds: int) -> str:
    """
    Formats a time duration given in seconds into a human-readable string.

    Args:
        seconds (int): The time duration in seconds.

    Returns:
        str: The formatted time string.
    """
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    elif s < 86400:
        return f"{s // 3600}h {s // 60 % 60:02d}m {s % 60:02d}s"
    else:
        return f"{s // 86400}d {s // 3600 % 24:02d}h {s // 60 % 60:02d}m"
