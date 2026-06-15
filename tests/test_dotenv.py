import os
import tempfile
import unittest
from pathlib import Path

from briefly.dotenv import load_dotenv


class TestDotenv(unittest.TestCase):
    def test_parses_and_does_not_override_real_env(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / ".env"
            p.write_text(
                "# a comment\n"
                "\n"
                "BRIEFLY_DIARIZE_URL=https://d.example/diarize\n"
                'export BRIEFLY_WHISPER_HOST="whisper.local"\n'
                "BRIEFLY_WHISPER_PORT=10300\n"
                "BRIEFLY_ALREADY_SET=fromfile\n"
            )
            os.environ.pop("BRIEFLY_DIARIZE_URL", None)
            os.environ.pop("BRIEFLY_WHISPER_HOST", None)
            os.environ.pop("BRIEFLY_WHISPER_PORT", None)
            os.environ["BRIEFLY_ALREADY_SET"] = "fromenv"
            try:
                loaded = load_dotenv(p)
                self.assertEqual(os.environ["BRIEFLY_DIARIZE_URL"], "https://d.example/diarize")
                self.assertEqual(os.environ["BRIEFLY_WHISPER_HOST"], "whisper.local")  # quotes + export
                self.assertEqual(os.environ["BRIEFLY_WHISPER_PORT"], "10300")
                self.assertEqual(os.environ["BRIEFLY_ALREADY_SET"], "fromenv")          # not overridden
                self.assertNotIn("BRIEFLY_ALREADY_SET", loaded)
            finally:
                for k in ("BRIEFLY_DIARIZE_URL", "BRIEFLY_WHISPER_HOST", "BRIEFLY_WHISPER_PORT",
                          "BRIEFLY_ALREADY_SET"):
                    os.environ.pop(k, None)

    def test_missing_file_is_noop(self):
        self.assertEqual(load_dotenv("/nope/.env"), {})


if __name__ == "__main__":
    unittest.main()
