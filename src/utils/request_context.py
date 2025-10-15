from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

_thread_id_var: ContextVar[Optional[str]] = ContextVar("thread_id", default=None)
_user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


def set_thread_id(thread_id: Optional[str]) -> Optional[Token]:
    if thread_id is None:
        return None
    return _thread_id_var.set(thread_id)


def reset_thread_id(token: Optional[Token]) -> None:
    if token is not None:
        _thread_id_var.reset(token)


def get_thread_id() -> Optional[str]:
    return _thread_id_var.get()


def set_user_id(user_id: Optional[str]) -> Optional[Token]:
    if user_id is None:
        return None
    return _user_id_var.set(user_id)


def reset_user_id(token: Optional[Token]) -> None:
    if token is not None:
        _user_id_var.reset(token)


def get_user_id() -> Optional[str]:
    return _user_id_var.get()
