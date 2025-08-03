import time
from src.portfolio_backtester.utils.timeout import TimeoutManager


def test_timeout_none_returns_false():
    tm = TimeoutManager(timeout_seconds=None)
    assert tm.check_timeout() is False


def test_timeout_not_elapsed():
    tm = TimeoutManager(timeout_seconds=5)
    # Immediately check, should not timeout
    assert tm.check_timeout() is False


def test_timeout_elapsed():
    tm = TimeoutManager(timeout_seconds=0.1)
    time.sleep(0.15)
    assert tm.check_timeout() is True


def test_timeout_reset():
    tm = TimeoutManager(timeout_seconds=0.1)
    time.sleep(0.15)
    # First check should indicate timeout
    assert tm.check_timeout() is True
    # Reset timer, then check again immediately should be False
    tm.reset()
    assert tm.check_timeout() is False


def test_timeout_invalid_value():
    tm = TimeoutManager(timeout_seconds="invalid")
    assert tm.check_timeout() is False
