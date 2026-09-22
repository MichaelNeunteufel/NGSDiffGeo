"""Assemble the persistent GitHub Pages tree from a Sphinx HTML build.

The site directory is a generated branch: main updates ``dev/`` and a release
tag updates both ``vX.Y.Z/`` and the root (the latest stable release).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


VERSION = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")
ROOT_MANIFEST = ".release-root-entries.json"
RESERVED = {"dev", "versions.json", ".nojekyll", ROOT_MANIFEST, "CNAME"}


def _version_key(value: str) -> tuple[int, int, int]:
    return tuple(map(int, VERSION.fullmatch(value).groups()))


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _copy_contents(source: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    names = []
    for item in source.iterdir():
        if item.name in RESERVED or VERSION.fullmatch(item.name) or item.name == ".git":
            raise ValueError(f"Sphinx output contains a reserved top-level name: {item.name}")
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        names.append(item.name)
    return sorted(names)


def _pages(directory: Path) -> list[str]:
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*.html"))


def _write_versions(site: Path) -> None:
    stable = None
    manifest = site / ROOT_MANIFEST
    if manifest.exists():
        stable = json.loads(manifest.read_text(encoding="utf-8")).get("version")

    versions = []
    root_pages = _pages(site)
    # Exclude nested versions from the root-page list.
    root_pages = [p for p in root_pages if p.split("/", 1)[0] != "dev" and not VERSION.fullmatch(p.split("/", 1)[0])]
    if root_pages:
        versions.append({"path": "", "label": f"Latest release ({stable})" if stable else "Current documentation", "pages": root_pages})
    if (site / "dev").is_dir():
        versions.append({"path": "dev/", "label": "Development", "pages": _pages(site / "dev")})
    releases = sorted(
        (p for p in site.iterdir() if p.is_dir() and VERSION.fullmatch(p.name)),
        key=lambda p: _version_key(p.name),
        reverse=True,
    )
    for release in releases:
        versions.append({"path": release.name + "/", "label": release.name, "pages": _pages(release)})
    (site / "versions.json").write_text(json.dumps({"versions": versions}, indent=2) + "\n", encoding="utf-8")
    (site / ".nojekyll").touch()


def update_site(site: Path, build: Path, channel: str, version: str | None = None, bootstrap_root: Path | None = None) -> None:
    site = site.resolve()
    build = build.resolve()
    if not build.is_dir() or not (build / "index.html").is_file():
        raise ValueError(f"Not a Sphinx HTML build: {build}")
    if site == build or site in build.parents or build in site.parents:
        raise ValueError("Site and build directories must be separate")
    if channel not in {"dev", "release"}:
        raise ValueError("Channel must be dev or release")
    if channel == "release" and (not version or not VERSION.fullmatch(version)):
        raise ValueError("Release version must be a final tag such as v0.3.0")
    if channel == "dev" and version is not None:
        raise ValueError("Development builds must not have a release version")
    if bootstrap_root is not None:
        bootstrap_root = bootstrap_root.resolve()
        if channel != "dev" or (site.exists() and any(site.iterdir())):
            raise ValueError("Bootstrap is only allowed for an empty development site")
        if not bootstrap_root.is_dir() or not (bootstrap_root / "index.html").is_file():
            raise ValueError(f"Not a Sphinx HTML bootstrap build: {bootstrap_root}")
        if bootstrap_root == site or bootstrap_root in site.parents or site in bootstrap_root.parents:
            raise ValueError("Bootstrap and site directories must be separate")

    site.mkdir(parents=True, exist_ok=True)
    if bootstrap_root is not None:
        _copy_contents(bootstrap_root, site)
    if channel == "dev":
        _remove(site / "dev")
        _copy_contents(build, site / "dev")
    else:
        snapshot = site / version
        _remove(snapshot)
        _copy_contents(build, snapshot)
        manifest = site / ROOT_MANIFEST
        previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None
        if previous is None or _version_key(version) >= _version_key(previous["version"]):
            if previous is not None:
                old_names = previous["entries"]
            else:
                old_names = [p.name for p in site.iterdir() if p.name not in RESERVED and not VERSION.fullmatch(p.name) and p.name != ".git"]
            for name in old_names:
                if name in RESERVED or VERSION.fullmatch(name) or name == ".git" or Path(name).name != name:
                    raise ValueError(f"Invalid root manifest entry: {name}")
                _remove(site / name)
            names = _copy_contents(build, site)
            manifest.write_text(json.dumps({"version": version, "entries": names}, indent=2) + "\n", encoding="utf-8")
    _write_versions(site)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--channel", choices=["dev", "release"], required=True)
    parser.add_argument("--version")
    parser.add_argument("--bootstrap-root", type=Path)
    args = parser.parse_args()
    update_site(args.site_dir, args.build_dir, args.channel, args.version, args.bootstrap_root)


if __name__ == "__main__":
    main()
