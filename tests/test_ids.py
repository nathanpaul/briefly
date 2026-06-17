import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from briefly.ids import is_ulid, new_ulid, next_meeting_id


class TestNextMeetingId(unittest.TestCase):
    def test_first_id_is_0001(self):
        with TemporaryDirectory() as td:
            self.assertEqual(next_meeting_id(td), "meeting_0001")

    def test_increments_past_highest(self):
        with TemporaryDirectory() as td:
            for name in ("meeting_0001", "meeting_0003", "not-a-meeting", "meeting_x"):
                (Path(td) / name).mkdir()
            self.assertEqual(next_meeting_id(td), "meeting_0004")  # max(1,3)+1, ignores non-matches

    def test_custom_prefix(self):
        with TemporaryDirectory() as td:
            (Path(td) / "sync_0007").mkdir()
            self.assertEqual(next_meeting_id(td, prefix="sync_"), "sync_0008")

    def test_missing_dir_starts_at_one(self):
        self.assertEqual(next_meeting_id("/no/such/dir/here"), "meeting_0001")


class TestUlid(unittest.TestCase):
    def test_length_and_charset(self):
        u = new_ulid()
        self.assertEqual(len(u), 26)
        self.assertTrue(is_ulid(u))

    def test_deterministic_with_injected_args(self):
        u = new_ulid(timestamp_ms=0, randomness=b"\x00" * 10)
        self.assertEqual(u, "0" * 26)

    def test_time_ordering(self):
        earlier = new_ulid(timestamp_ms=1_000, randomness=b"\xff" * 10)
        later = new_ulid(timestamp_ms=2_000, randomness=b"\x00" * 10)
        self.assertLess(earlier, later)  # lexicographic == chronological

    def test_randomness_length_validated(self):
        with self.assertRaises(ValueError):
            new_ulid(randomness=b"\x00")

    def test_is_ulid_rejects_bad(self):
        self.assertFalse(is_ulid("short"))
        self.assertFalse(is_ulid("I" * 26))  # I is not in Crockford alphabet


if __name__ == "__main__":
    unittest.main()
