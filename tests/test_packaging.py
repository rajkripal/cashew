"""Regression test for the packaging manifest.

Issue #7: `extractors/` existed in the repo with an `__init__.py` but was
missing from `[tool.setuptools].packages`, so `pip install cashew-brain`
shipped a wheel without it and crashed on first import.

This test asserts the manifest declares every top-level importable
package directory, so a contributor adding a new package folder can't
silently forget to list it.
"""
from __future__ import annotations

from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found]


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Top-level directories that live next to cashew_cli.py but are not
# shipped Python packages (tests, docs, developer scripts, vendored tools).
EXCLUDED_DIRS = {
    "tests",
    "docs",
    "dist",
    "build",
    "bench",
    "cron",
    "data",
    "models",
    "prompts",
    "templates",
    "examples",
    ".github",
    ".pytest_cache",
    ".venv",
    "venv",
}


def _pyproject_packages() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    return list(data["tool"]["setuptools"]["packages"])


def _top_level_package_dirs() -> list[str]:
    out = []
    for entry in REPO_ROOT.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") and entry.name not in EXCLUDED_DIRS:
            continue
        if entry.name in EXCLUDED_DIRS:
            continue
        if (entry / "__init__.py").exists():
            out.append(entry.name)
    return sorted(out)


def test_pyproject_declares_every_top_level_package():
    declared = set(_pyproject_packages())
    on_disk = set(_top_level_package_dirs())
    missing = on_disk - declared
    assert not missing, (
        f"pyproject.toml [tool.setuptools].packages is missing importable "
        f"top-level packages: {sorted(missing)}. If a package isn't meant "
        f"to ship, add its directory to EXCLUDED_DIRS in this test."
    )


def test_extractors_is_declared():
    """Explicit pin for the exact regression in issue #7."""
    assert "extractors" in _pyproject_packages()
