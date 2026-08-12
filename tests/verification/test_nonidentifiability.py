from __future__ import annotations

import builtins
from typing import Any

import pytest

from hiprof import HPFalsifier


def test_check_none_without_ananke_raises_informative_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        glob: Any = None,
        loc: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "ananke" or name.startswith("ananke."):
            raise ModuleNotFoundError(
                "No module named 'ananke'",
                name="ananke",
            )

        return real_import(name, glob, loc, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    falsifier = HPFalsifier(
        "X -> Y; X <-> Y",
        treatments="X",
        outcomes="Y",
    )

    with pytest.raises(
        ImportError,
        match=r"hiprof\[nonidentifiability\]",
    ):
        falsifier.check(None)
