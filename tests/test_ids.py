import unittest

from briefly.ids import is_ulid, new_ulid


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
