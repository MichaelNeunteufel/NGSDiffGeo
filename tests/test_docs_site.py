"""Tests for the generated, versioned GitHub Pages layout (no NGSolve needed)."""

import json
import tempfile
import unittest
from pathlib import Path
from sys import path

path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from update_docs_site import update_site  # noqa: E402


class TestDocsSite(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.site = self.base / "site"
        self.build = self.base / "build"
        self.old = self.base / "old"
        for directory, title in ((self.build, "development"), (self.old, "old root")):
            (directory / "tutorials").mkdir(parents=True)
            (directory / "index.html").write_text(title)
            (directory / "tutorials" / "example.html").write_text(title)

    def manifest(self):
        return json.loads((self.site / "versions.json").read_text())["versions"]

    def test_bootstrap_and_release_preserve_development(self):
        update_site(self.site, self.build, "dev", bootstrap_root=self.old)
        self.assertEqual((self.site / "index.html").read_text(), "old root")
        self.assertEqual((self.site / "dev" / "index.html").read_text(), "development")
        self.assertEqual([v["path"] for v in self.manifest()], ["", "dev/"])

        (self.build / "index.html").write_text("release")
        (self.build / "tutorials" / "example.html").unlink()
        update_site(self.site, self.build, "release", "v0.3.0")
        self.assertEqual((self.site / "index.html").read_text(), "release")
        self.assertEqual((self.site / "v0.3.0" / "index.html").read_text(), "release")
        self.assertFalse((self.site / "tutorials" / "example.html").exists())
        self.assertEqual((self.site / "dev" / "index.html").read_text(), "development")
        entries = self.manifest()
        self.assertEqual([v["path"] for v in entries], ["", "dev/", "v0.3.0/"])
        self.assertEqual(entries[0]["label"], "Latest release (v0.3.0)")
        self.assertNotIn("tutorials/example.html", entries[0]["pages"])

        (self.build / "index.html").write_text("new development")
        update_site(self.site, self.build, "dev")
        self.assertEqual((self.site / "index.html").read_text(), "release")
        self.assertEqual((self.site / "dev" / "index.html").read_text(), "new development")

        update_site(self.site, self.build, "release", "v0.4.0")
        (self.build / "index.html").write_text("old release rerun")
        update_site(self.site, self.build, "release", "v0.3.0")
        self.assertEqual((self.site / "index.html").read_text(), "new development")
        self.assertEqual((self.site / "v0.3.0" / "index.html").read_text(), "old release rerun")
        self.assertEqual(self.manifest()[0]["label"], "Latest release (v0.4.0)")

    def test_rejects_invalid_version_and_overlap(self):
        with self.assertRaises(ValueError):
            update_site(self.site, self.build, "release", "v0.3.0rc1")
        with self.assertRaises(ValueError):
            update_site(self.build, self.build, "dev")
        self.assertFalse(self.site.exists())


if __name__ == "__main__":
    unittest.main()
