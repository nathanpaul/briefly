import tempfile
import unittest
from pathlib import Path

from briefly.state import read_last_meeting, write_last_meeting

MID = "01J9ZC8Q9F7Y3K2N5R6T8W0X1Z"


class TestState(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            rec = Path(td) / "recordings"
            write_last_meeting(rec, MID)
            self.assertEqual(read_last_meeting(rec), MID)
            self.assertEqual((rec / ".last-meeting-id").read_text().strip(), MID)

    def test_missing_is_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(read_last_meeting(Path(td) / "recordings"))

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            rec = Path(td) / "a" / "b" / "recordings"
            write_last_meeting(rec, MID)
            self.assertEqual(read_last_meeting(rec), MID)


if __name__ == "__main__":
    unittest.main()
