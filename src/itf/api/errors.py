"""R4 — an error says WHY and HOW TO FIX IT.

Inherited from the sibling project, where it is an explicit rule, and it is the
difference between an API you can use and one you have to guess at.

`code` is a CONTRACT: a stable slug the UI may switch on. `message` and `hint`
are for people. The hint is not fine print -- it is the half that says what to do
next, and an error without one makes the reader reverse-engineer the fix.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def problem(status: int, code: str, message: str, hint: str | None = None, **extra: Any) -> HTTPException:
    detail: dict[str, Any] = {"code": code, "message": message}
    if hint:
        detail["hint"] = hint
    detail.update(extra)
    return HTTPException(status_code=status, detail=detail)


def not_found(code: str, message: str, hint: str | None = None, **extra: Any) -> HTTPException:
    return problem(404, code, message, hint, **extra)


def conflict(code: str, message: str, hint: str | None = None, **extra: Any) -> HTTPException:
    """409 — clashes with the state: in use, already exists, still running."""
    return problem(409, code, message, hint, **extra)


def bad_request(code: str, message: str, hint: str | None = None, **extra: Any) -> HTTPException:
    """400 — the request is impossible, and the body says why.

    Validated BEFORE, never during: a 400 on the way in is worth a thousand stack
    traces inside a job thread half an hour later.
    """
    return problem(400, code, message, hint, **extra)
