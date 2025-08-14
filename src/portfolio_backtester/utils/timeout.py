import logging
import time

logger = logging.getLogger(__name__)


class TimeoutManager:
    def __init__(self, timeout_seconds: float | None, start_time: float | None = None) -> None:
        self.timeout = timeout_seconds
        # Use provided start_time (wall-clock, float), or now if not given
        self.start_time = start_time if start_time is not None else time.time()

    def check_timeout(self) -> bool:
        if self.timeout is None:
            return False
        try:
            timeout_value = float(self.timeout) if self.timeout is not None else None
            if timeout_value is None:
                return False
            elapsed_time = time.time() - self.start_time
            if elapsed_time > timeout_value:
                logger.warning(
                    f"Timeout of {timeout_value} seconds exceeded. Elapsed time: {elapsed_time:.2f} seconds."
                )
                print(f"Warning: Timeout of {timeout_value} seconds exceeded.")
                return True
        except (TypeError, ValueError, AttributeError):
            return False
        return False

    def reset(self, new_start_time: float | None = None) -> None:
        """Reset the timeout timer. Optionally set a new start_time (float)."""
        self.start_time = new_start_time if new_start_time is not None else time.time()
