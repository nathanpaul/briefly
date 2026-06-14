import unittest

from briefly.audio.devices import (
    find_input,
    is_clipping,
    is_flat_zero,
    parse_audio_inputs,
    parse_levels,
)

LIST_FIXTURE = """\
[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] FaceTime HD Camera
[AVFoundation indev @ 0x1] [1] Capture screen 0
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] Cubilux CB5 MIC2
[AVFoundation indev @ 0x1] [1] MacBook Pro Microphone
[AVFoundation indev @ 0x1] [2] Cubilux CB5 Line In
"""

VOLUMEDETECT_FIXTURE = """\
[Parsed_volumedetect_0 @ 0x1] n_samples: 192000
[Parsed_volumedetect_0 @ 0x1] mean_volume: -29.6 dB
[Parsed_volumedetect_0 @ 0x1] max_volume: -1.3 dB
"""


class TestDeviceParsing(unittest.TestCase):
    def test_parse_audio_inputs_ignores_video(self):
        inputs = parse_audio_inputs(LIST_FIXTURE)
        self.assertEqual([i.name for i in inputs],
                         ["Cubilux CB5 MIC2", "MacBook Pro Microphone", "Cubilux CB5 Line In"])
        self.assertEqual([i.index for i in inputs], [0, 1, 2])

    def test_find_input_by_name(self):
        inputs = parse_audio_inputs(LIST_FIXTURE)
        self.assertEqual(find_input("Cubilux CB5 Line In", inputs).index, 2)
        self.assertIsNone(find_input("Nonexistent", inputs))

    def test_parse_levels(self):
        self.assertEqual(parse_levels(VOLUMEDETECT_FIXTURE), (-29.6, -1.3))
        self.assertEqual(parse_levels("no levels here"), (None, None))


class TestClassifiers(unittest.TestCase):
    def test_flat_zero_is_wrong_device(self):
        self.assertTrue(is_flat_zero(-91.0, -91.0))
        self.assertFalse(is_flat_zero(-90.3, -84.3))  # real input fluctuates
        self.assertFalse(is_flat_zero(None, -91.0))

    def test_clipping_threshold(self):
        self.assertTrue(is_clipping(0.0, -0.1))
        self.assertTrue(is_clipping(-0.1, -0.1))
        self.assertFalse(is_clipping(-6.0, -0.1))
        self.assertFalse(is_clipping(None, -0.1))


if __name__ == "__main__":
    unittest.main()
