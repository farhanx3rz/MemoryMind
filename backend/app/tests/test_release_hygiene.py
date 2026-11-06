"""Unit and integration tests for release hygiene and sanitized distribution packaging.

Verifies:
1. should_exclude accurately detects and excludes:
   - Secret files (.env, backend/.env)
   - Python bytecode and compiled files (*.pyc, *.pyd, __pycache__)
   - Test and coverage caches (.pytest_cache, .coverage, htmlcov)
   - Local vector database and SQLite FTS indexes (chroma, *.sqlite3*)
   - Runtime jobs persistence and temporary report files (data/jobs, backend/data/jobs)
   - Log files (*.log)
   - Editor metadata (.vscode, .idea)
2. Preserves non-sensitive template/documentation files (.env.example) and source code.
3. Archive creation test: generates ZIP and verifies no prohibited file leaks into the archive.
"""

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

# Add backend directory to sys.path so scripts can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.package_dist import build_dist_archive, should_exclude


class ReleaseHygieneTests(unittest.TestCase):
    def test_should_exclude_secrets(self):
        self.assertTrue(should_exclude(".env"))
        self.assertTrue(should_exclude("backend/.env"))
        self.assertTrue(should_exclude(".env.production"))
        self.assertTrue(should_exclude("backend/.env.local"))

    def test_should_exclude_caches_and_bytecode(self):
        self.assertTrue(should_exclude("__pycache__"))
        self.assertTrue(should_exclude("app/__pycache__/main.cpython-312.pyc"))
        self.assertTrue(should_exclude(".pytest_cache/v/cache/nodeids"))
        self.assertTrue(should_exclude(".coverage"))
        self.assertTrue(should_exclude("htmlcov/index.html"))

    def test_should_exclude_runtime_databases_and_jobs(self):
        self.assertTrue(should_exclude("data/chroma/chroma.sqlite3"))
        self.assertTrue(should_exclude("backend/data/chroma/index"))
        self.assertTrue(should_exclude("data/memorymind.sqlite3"))
        self.assertTrue(should_exclude("backend/data/database.sqlite3-wal"))
        self.assertTrue(should_exclude("backend/data/jobs/job_123.json"))
        self.assertTrue(should_exclude("data/jobs/summary_456.pdf"))

    def test_should_exclude_logs_and_editor_artifacts(self):
        self.assertTrue(should_exclude("server.log"))
        self.assertTrue(should_exclude("backend/app.log"))
        self.assertTrue(should_exclude(".vscode/settings.json"))
        self.assertTrue(should_exclude(".idea/workspace.xml"))

    def test_preserves_legitimate_source_and_examples(self):
        self.assertFalse(should_exclude("backend/app/main.py"))
        self.assertFalse(should_exclude("backend/app/config.py"))
        self.assertFalse(should_exclude("extension/manifest.json"))
        self.assertFalse(should_exclude("extension/popup/popup.js"))
        self.assertFalse(should_exclude("README.md"))

    def test_build_dist_archive_excludes_secrets_and_artifacts(self):
        """Build an archive from a temporary directory tree and inspect every zip entry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "project"
            root.mkdir()

            # Create legitimate files
            (root / "README.md").write_text("Documentation", encoding="utf-8")
            (root / ".env.example").write_text("API_KEY=sample", encoding="utf-8")
            src_dir = root / "backend" / "app"
            src_dir.mkdir(parents=True)
            (src_dir / "main.py").write_text("# main code", encoding="utf-8")

            # Create files that MUST be excluded
            (root / ".env").write_text("SECRET_API_KEY=leak", encoding="utf-8")
            (root / "backend" / ".env").write_text("ANOTHER_SECRET=leak", encoding="utf-8")
            (root / "app.log").write_text("log data", encoding="utf-8")

            cache_dir = src_dir / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "main.cpython-312.pyc").write_bytes(b"\x00\x00")

            data_jobs_dir = root / "backend" / "data" / "jobs"
            data_jobs_dir.mkdir(parents=True)
            (data_jobs_dir / "job_secret.json").write_text("{}", encoding="utf-8")
            (data_jobs_dir / "report.pdf").write_bytes(b"%PDF-1.4")

            chroma_dir = root / "backend" / "data" / "chroma"
            chroma_dir.mkdir(parents=True)
            (chroma_dir / "chroma.sqlite3").write_bytes(b"sqlite data")

            # Build archive
            out_zip = Path(tmp_dir) / "dist" / "release.zip"
            build_dist_archive(root, out_zip)

            self.assertTrue(out_zip.exists())

            # Read back all entries in the ZIP
            with zipfile.ZipFile(out_zip, "r") as zf:
                namelist = zf.namelist()

                # Verify clean files are present
                self.assertIn("README.md", namelist)
                self.assertIn(".env.example", namelist)
                self.assertIn("backend/app/main.py", namelist)

                # Verify no secret or transient files leaked
                for name in namelist:
                    if name.endswith(".example"):
                        continue
                    self.assertFalse(Path(name).name.startswith(".env"))
                    self.assertFalse(name.endswith(".pyc"))
                    self.assertNotIn("__pycache__", name)
                    self.assertNotIn("jobs", name)
                    self.assertNotIn("chroma", name)
                    self.assertNotIn(".sqlite3", name)
                    self.assertFalse(name.endswith(".log"))


if __name__ == "__main__":
    unittest.main()
