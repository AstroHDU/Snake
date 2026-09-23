"""Explicit per-call uncertainty mode; never inferred from missing columns."""
from contextvars import ContextVar
from functools import wraps

ENABLED = ContextVar('sri_use_uncertainty', default=True)

def scoped(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        mode = kwargs.get('use_uncertainty', True)
        if not isinstance(mode, bool):
            raise ValueError('use_uncertainty must be boolean')
        token = ENABLED.set(mode)
        try:
            return function(*args, **kwargs)
        finally:
            ENABLED.reset(token)
    return wrapped
