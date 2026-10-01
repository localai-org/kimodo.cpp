#!/usr/bin/env python3
"""Restart-coverage tests for docker/entrypoint.sh weight completeness.

Runs the real entrypoint against a sandbox data directory. KIMODO_APP_DIR
points at stub downloader/server scripts that record their arguments, so
each case asserts which weights trigger a fetch and which flags the
downloader receives. Covers the partial first start (packed weights
present, sibling tokenizer.gguf missing) that used to leave a restarted
container unable to generate, and the KIMODO_TEXT_QUANTIZATION opt-in on
legacy-only volumes.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "docker" / "entrypoint.sh"

MOTION = "models/kimodo-soma-rp-v1.1-f32.gguf"
PACKED_Q8 = "Llama-3-Kimodo-Q8_0.gguf"
PACKED_Q4KM = "Llama-3-Kimodo-Q4_K_M.gguf"
TOKENIZER = "tokenizer.gguf"
LEGACY = "generated/llm2vec-text-bundle/layer-31.gguf"


class EntrypointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name) / "data"
        self.app = Path(self.temp.name) / "app"
        (self.app / "scripts").mkdir(parents=True)
        (self.app / "bin").mkdir(parents=True)
        self.downloader_log = self.temp.name + "/downloader.log"
        self.demo_log = self.temp.name + "/demo.log"

        # The entrypoint invokes the downloader via python3, so the stub
        # must be a python script; the demo is exec'd directly.
        downloader = self.app / "scripts" / "download_gguf_weights.py"
        downloader.write_text(
            "import sys\n"
            f"open({os.fspath(self.downloader_log)!r}, 'w')"
            ".write('\\n'.join(sys.argv[1:]) + '\\n')\n"
        )
        demo = self.app / "bin" / "kimodo-demo"
        demo.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$@\" > {os.fspath(self.demo_log)!r}\n"
        )
        demo.chmod(0o755)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, rel_path: str) -> Path:
        path = self.data / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00")
        return path

    def run_entrypoint(self, **env_overrides: str) -> subprocess.CompletedProcess:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "KIMODO_DATA": str(self.data),
            "KIMODO_APP_DIR": str(self.app),
            "KIMODO_MODELS": "soma-rp-v1.1",
        }
        env.update(env_overrides)
        return subprocess.run(
            ["sh", str(ENTRYPOINT)], env=env, capture_output=True, text=True
        )

    def downloader_args(self) -> list[str] | None:
        log = Path(self.downloader_log)
        if not log.is_file():
            return None
        return log.read_text().splitlines()

    def assert_demo_ran(self, ran: bool) -> None:
        if ran:
            self.assertTrue(Path(self.demo_log).is_file())
        else:
            self.assertFalse(Path(self.demo_log).exists())

    def test_entrypoint_parses(self) -> None:
        subprocess.run(["sh", "-n", str(ENTRYPOINT)], check=True)

    def test_no_models_skips_download(self) -> None:
        result = self.run_entrypoint(KIMODO_MODELS="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(self.downloader_args())
        self.assert_demo_ran(ran=True)

    def test_complete_packed_bundle_skips_download(self) -> None:
        self.write(MOTION)
        self.write(PACKED_Q8)
        self.write(TOKENIZER)
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(self.downloader_args())
        self.assert_demo_ran(ran=True)

    def test_legacy_bundle_skips_download_by_default(self) -> None:
        self.write(MOTION)
        self.write(LEGACY)
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(self.downloader_args())
        self.assert_demo_ran(ran=True)

    def test_packed_weights_without_tokenizer_resume_text_download(self) -> None:
        # Interrupted first start: packed weights landed before the
        # tokenizer. The restart must re-enter the downloader for text
        # instead of starting a server whose packed bundles are all
        # unavailable.
        self.write(MOTION)
        self.write(PACKED_Q8)
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.downloader_args()
        self.assertIsNotNone(args, "downloader must run for the missing tokenizer")
        assert args
        self.assertIn("--text-quantization", args)
        self.assertIn("q8_0", args)
        self.assertNotIn("--motion-only", args)
        self.assertIn("tokenizer.gguf missing", result.stderr)
        self.assert_demo_ran(ran=True)

    def test_explicit_quantization_fetches_packed_over_legacy(self) -> None:
        # Legacy-only volume: setting KIMODO_TEXT_QUANTIZATION opts in to
        # fetching that packed bundle on the next start.
        self.write(MOTION)
        self.write(LEGACY)
        result = self.run_entrypoint(KIMODO_TEXT_QUANTIZATION="q8_0")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.downloader_args()
        self.assertIsNotNone(args)
        assert args
        self.assertIn("--text-quantization", args)
        self.assertIn("q8_0", args)
        self.assertNotIn("--motion-only", args)
        self.assert_demo_ran(ran=True)

    def test_explicit_quantization_fetches_second_variant(self) -> None:
        # A complete default bundle is reused, but an explicit request for
        # another variant still fetches it.
        self.write(MOTION)
        self.write(PACKED_Q8)
        self.write(TOKENIZER)
        result = self.run_entrypoint(KIMODO_TEXT_QUANTIZATION="q4_k_m")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.downloader_args()
        self.assertIsNotNone(args)
        assert args
        self.assertIn("--text-quantization", args)
        self.assertIn("q4_k_m", args)
        self.assertNotIn("--motion-only", args)

    def test_complete_text_with_missing_motion_is_motion_only(self) -> None:
        self.write(PACKED_Q8)
        self.write(TOKENIZER)
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.downloader_args()
        self.assertIsNotNone(args)
        assert args
        self.assertIn("--motion-only", args)
        self.assertNotIn("--text-quantization", args)

    def test_unknown_quantization_fails_fast(self) -> None:
        self.write(MOTION)
        self.write(LEGACY)
        result = self.run_entrypoint(KIMODO_TEXT_QUANTIZATION="q9_typo")
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(self.downloader_args())
        self.assertIn("unknown KIMODO_TEXT_QUANTIZATION", result.stderr)


if __name__ == "__main__":
    unittest.main()
