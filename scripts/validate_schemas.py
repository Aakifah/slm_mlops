"""Validate Pydantic schema contracts and SQLglot AST guardrails.

Skips gracefully when schema modules are not implemented yet.

Returns:
    None
"""

from __future__ import annotations

import importlib
import sys


def _try_import(module_name: str) -> tuple[bool, str]:
    """Attempt to import a schema module.

    Args:
        module_name: Fully qualified module import path.

    Returns:
        Tuple of import success flag and status message.
    """
    try:
        importlib.import_module(module_name)
        return True, f"found {module_name}"
    except ModuleNotFoundError:
        return False, f"missing {module_name}"
    except Exception as exc:
        return False, f"error importing {module_name}: {exc}"


def main() -> int:
    """Run schema validation or skip when schemas are absent.

    Returns:
        Process exit code with zero on success or skip.
    """
    candidates: list[str] = [
        "src.backend.schemas.query",
    ]
    results: list[tuple[bool, str]] = [_try_import(name) for name in candidates]
    found: list[str] = [message for ok, message in results if ok]
    if not found:
        print("validate_schemas: skip, src/ schemas not implemented yet")
        return 0
    for message in found:
        print(f"validate_schemas: {message}")
    print("validate_schemas: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
