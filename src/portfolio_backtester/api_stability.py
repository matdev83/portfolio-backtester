import functools
from typing import Any, Callable, TypeVar, cast

# Define a type variable for the decorated function
F = TypeVar("F", bound=Callable[..., Any])


def api_stable(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
