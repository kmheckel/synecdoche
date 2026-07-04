"""Default backend management.

Transforms resolve their backend at call time: an explicit ``backend=``
argument wins, otherwise the module default set by ``configure()`` is used.
Late binding matters — ``@syn.jit`` runs at import time, usually before any
backend exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .exceptions import FrameworkError

if TYPE_CHECKING:
    from .backend import Backend

_default: Backend | None = None


def configure(**kwargs: Any) -> Backend:
    """Create a ``Backend`` from the given options and install it as the default.

    Accepts exactly the ``Backend`` constructor arguments (model, mcp,
    archive, sandbox, heal, ...). Returns the backend so it can also be
    passed explicitly to individual transforms.
    """
    from .backend import Backend

    global _default
    _default = Backend(**kwargs)
    return _default


def set_default_backend(backend: Backend | None) -> None:
    """Install (or clear) the default backend directly."""
    global _default
    _default = backend


def current_backend() -> Backend:
    if _default is None:
        raise FrameworkError(
            "No default backend configured. Call synecdoche.configure(model=...) "
            "once at startup, or pass backend= to the transform."
        )
    return _default


__all__ = ["configure", "current_backend", "set_default_backend"]
