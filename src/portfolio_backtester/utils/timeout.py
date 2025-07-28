import logging
import time

logger = logging.getLogger(__name__)


class TimeoutManager:
    def __init__(self, timeout_seconds):
        self.timeout = timeout_seconds
        self.start_time = time.time()

    def check_timeout(self):
        if self.timeout is None:
            return False
        
        try:
            timeout_value = float(self.timeout) if self.timeout is not None else None
            if timeout_value is None:
                return False
                
            elapsed_time = time.time() - self.start_time
            if elapsed_time > timeout_value:
                logger.warning(f"Timeout of {timeout_value} seconds exceeded. Elapsed time: {elapsed_time:.2f} seconds.")
                print(f"Warning: Timeout of {timeout_value} seconds exceeded.")
                return True
        except (TypeError, ValueError, AttributeError):
            return False
        
        return False
