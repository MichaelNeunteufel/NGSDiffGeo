"""Shared measurement and provenance helpers for benchmark programs."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
from contextlib import nullcontext
from pathlib import Path
import statistics
import subprocess
import sys


def _statistics(samples):
    median = statistics.median(samples)
    median_absolute_deviation = statistics.median(
        abs(sample - median) for sample in samples
    )
    return {
        "minimum_seconds": min(samples),
        "median_seconds": median,
        "median_absolute_deviation_seconds": median_absolute_deviation,
        "relative_median_absolute_deviation": (
            median_absolute_deviation / median if median else 0.0
        ),
        "samples_seconds": samples,
    }


def _alternating_order(names, iteration):
    return names if iteration % 2 == 0 else tuple(reversed(names))


def _file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
    }


def _module_identity(name):
    module = importlib.import_module(name)
    return _file_identity(module.__file__)


def _extension_identity():
    return _module_identity("ngsdiffgeo.ngsdiffgeo")


def _wrapper_identity():
    return _module_identity("ngsdiffgeo.wrappers")


def _peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the other supported Unix CI systems
    # report KiB.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _retained_memory_sample(
    factory,
    count,
    *,
    count_key,
    per_item_key,
    metadata=None,
    context_factory=nullcontext,
):
    """Measure peak-RSS growth while keeping ``count`` factory results live."""

    metadata = dict(metadata or {})
    with context_factory():
        warm = factory()
    del warm
    gc.collect()
    before = _peak_rss_bytes()
    if before is None:
        return {
            "supported": False,
            **metadata,
            count_key: count,
            "reason": "resource.getrusage is unavailable",
        }

    with context_factory():
        retained = [factory() for _ in range(count)]
    after = _peak_rss_bytes()
    # Keep the references live until after the high-water mark is sampled.
    assert len(retained) == count
    delta = max(0, after - before)
    return {
        "supported": True,
        **metadata,
        count_key: count,
        "peak_rss_before_bytes": before,
        "peak_rss_after_bytes": after,
        "peak_rss_delta_bytes": delta,
        per_item_key: delta / count,
    }


def _source_manifest(root):
    """Hash every local file that can materially affect benchmark binaries."""

    root = Path(root).resolve()
    candidates = {
        root / "CMakeLists.txt",
        root / "ngsolve_addon.cmake",
        root / "pyproject.toml",
    }
    source_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".inc",
        ".py",
    }
    candidates.update(
        path
        for path in (root / "src").rglob("*")
        if path.is_file() and path.suffix in source_suffixes
    )
    candidates.update(
        path
        for path in (root / "benchmarks").glob("*.py")
        if path.is_file()
    )
    relative_paths = sorted(
        path.relative_to(root).as_posix()
        for path in candidates
        if path.is_file()
    )
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with (root / relative_path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return {
        "algorithm": "sha256(relative-path NUL content NUL)",
        "sha256": digest.hexdigest(),
        "file_count": len(relative_paths),
        "files": relative_paths,
    }


def _repository_identity():
    root = Path(__file__).resolve().parents[1]
    source_manifest = _source_manifest(root)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        source_status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *source_manifest["files"],
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        source_manifest["has_changes"] = None
        return {
            "revision": None,
            "tracked_changes": None,
            "tracked_diff_sha256": None,
            "source_manifest": source_manifest,
        }
    source_manifest["has_changes"] = bool(source_status.strip())
    return {
        "revision": revision,
        "tracked_changes": bool(tracked_status.strip()),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "source_manifest": source_manifest,
    }


def _subprocess_json(command):
    process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("benchmark worker produced no JSON output")
    return json.loads(lines[-1])
