"""Tests for the merge stage (stdlib unittest only — no network/hardware/3rd-party)."""
import json
import tempfile
import unittest
from pathlib import Path

from briefly.merge import (
    DiarDoc,
    InputError,
    MergeConfig,
    WhisperDoc,
    main,
    merge,
    run,
    text_similarity,
)
from briefly.models import MeetingManifest, SpeakersMap

MID = "01J9ZC8Q9F7Y3K2N5R6T8W0X1Z"


# --- fixtures -------------------------------------------------------------------------

def _manifest(partial: bool = False, mic_off: float = 0.0, line_off: float = 0.0) -> MeetingManifest:
    return MeetingManifest.from_dict({
        "meeting_id": MID,
        "date": "2026-06-14",
        "started_at": "2026-06-14T09:00:03Z",
        "ended_at": "2026-06-14T09:52:10Z",
        "partial": partial,
        "attendees": ["Jane Doe", "John Smith"],
        "capture": {
            "mode": "dual-process", "sample_rate": 48000, "format": "pcm_s16le",
            "channels": 2, "ffmpeg": "8.1.1", "offset_method": "process-start-delta",
        },
        "channels": {
            "mic": {"file": "mic.wav", "device_name": "mic", "speaker": "Me",
                    "start_offset_sec": mic_off},
            "line": {"file": "line.wav", "device_name": "line",
                     "start_offset_sec": line_off},
        },
    })


def _whisper(segments: list[dict], duration: float = 100.0) -> WhisperDoc:
    return WhisperDoc.from_dict({"language": "en", "duration_sec": duration,
                                 "segments": segments})


def _seg(i, start, end, text, words=None, avg_logprob=-0.1):
    d = {"id": i, "start": start, "end": end, "text": text, "avg_logprob": avg_logprob,
         "no_speech_prob": 0.01}
    if words is not None:
        d["words"] = words
    return d


def _diar(segments: list[tuple], duration: float = 100.0) -> DiarDoc:
    return DiarDoc.from_dict({
        "model": "pyannote/speaker-diarization-community-1",
        "duration_sec": duration,
        "num_speakers": len({s[0] for s in segments}),
        "segments": [{"speaker": sp, "start": st, "end": en} for sp, st, en in segments],
    })


# --- text similarity ------------------------------------------------------------------

class TestTextSimilarity(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(text_similarity("Keep it read only", "keep IT read only"), 1.0)

    def test_disjoint(self):
        self.assertEqual(text_similarity("alpha beta", "gamma delta"), 0.0)

    def test_partial(self):
        # tokens {a,b,c} vs {a,b,d}: inter 2 / union 4 = 0.5
        self.assertAlmostEqual(text_similarity("a b c", "a b d"), 0.5)


# --- clean two-speaker call -----------------------------------------------------------

class TestCleanTwoSpeaker(unittest.TestCase):
    def test_two_speakers_sorted_indexed(self):
        manifest = _manifest()
        mic = _whisper([_seg(0, 11.2, 13.0, "Let me walk through the phasing.")])
        line = _whisper([
            _seg(0, 13.4, 16.1, "Can we keep it read only?"),
            _seg(1, 20.0, 22.0, "I think we should defer the cutover."),
        ])
        diar = _diar([("SPEAKER_00", 13.0, 17.0), ("SPEAKER_01", 19.5, 23.0)])

        t = merge(manifest, mic, line, diar, generated_at="2026-06-14T10:05:00Z")

        # Three turns, contiguous indices, sorted by start.
        self.assertEqual([turn.index for turn in t.turns], [0, 1, 2])
        self.assertEqual([turn.start for turn in t.turns], [11.2, 13.4, 20.0])
        self.assertEqual(t.turns[0].speaker, "Me")
        self.assertEqual(t.turns[0].channel, "mic")
        self.assertIsNone(t.turns[0].diarization_confidence)
        # Two distinct line speakers numbered by first appearance.
        self.assertEqual(t.turns[1].speaker, "Speaker_1")
        self.assertEqual(t.turns[2].speaker, "Speaker_2")
        labels = {s.label for s in t.speakers}
        self.assertEqual(labels, {"Me", "Speaker_1", "Speaker_2"})
        # diarization confidence is a fraction in (0,1].
        self.assertGreater(t.turns[1].diarization_confidence, 0.0)
        self.assertLessEqual(t.turns[1].diarization_confidence, 1.0)
        # No spurious flags on a clean call.
        self.assertEqual(t.turns[1].flags, [])

    def test_start_offset_applied(self):
        manifest = _manifest(line_off=2.0)
        mic = _whisper([_seg(0, 1.0, 2.0, "hello there")])
        line = _whisper([_seg(0, 5.0, 7.0, "general kenobi")])
        diar = _diar([("SPEAKER_00", 6.0, 10.0)])  # diar is line-local too
        t = merge(manifest, mic, line, diar, generated_at="x")
        line_turn = [x for x in t.turns if x.channel == "line"][0]
        # 5.0 + 2.0 offset = 7.0
        self.assertEqual(line_turn.start, 7.0)
        self.assertEqual(line_turn.end, 9.0)


# --- word-level speaker split ---------------------------------------------------------

class TestWordLevelSplit(unittest.TestCase):
    def test_split_mid_segment(self):
        manifest = _manifest()
        words = [
            {"word": "Yes ", "start": 10.0, "end": 10.4, "prob": 0.99},
            {"word": "exactly ", "start": 10.4, "end": 10.9, "prob": 0.98},
            {"word": "no ", "start": 12.1, "end": 12.4, "prob": 0.97},
            {"word": "way", "start": 12.4, "end": 12.8, "prob": 0.96},
        ]
        mic = _whisper([])
        line = _whisper([_seg(0, 10.0, 12.8, "Yes exactly no way", words=words)])
        # Two speakers, one owns [10,11], the other [12,13].
        diar = _diar([("SPEAKER_00", 9.5, 11.0), ("SPEAKER_01", 12.0, 13.0)])

        t = merge(manifest, mic, line, diar, generated_at="x")
        line_turns = [x for x in t.turns if x.channel == "line"]
        self.assertEqual(len(line_turns), 2)
        self.assertEqual(line_turns[0].text, "Yes exactly")
        self.assertEqual(line_turns[0].speaker, "Speaker_1")
        self.assertEqual(line_turns[1].text, "no way")
        self.assertEqual(line_turns[1].speaker, "Speaker_2")
        # A run never mixes speakers.
        self.assertNotEqual(line_turns[0].speaker_id, line_turns[1].speaker_id)


# --- no diarization overlap (nearest + warning) ---------------------------------------

class TestNoOverlapFallback(unittest.TestCase):
    def test_nearest_within_window(self):
        manifest = _manifest()
        mic = _whisper([])
        # Segment [5.0,5.3] doesn't overlap diar [5.5,8.0] but is within 0.5s.
        line = _whisper([_seg(0, 5.0, 5.3, "quick aside")])
        diar = _diar([("SPEAKER_00", 5.5, 8.0)])
        t = merge(manifest, mic, line, diar, generated_at="x")
        lt = [x for x in t.turns if x.channel == "line"][0]
        self.assertEqual(lt.speaker, "Speaker_1")
        self.assertEqual(lt.diarization_confidence, 0.0)  # nearest, no overlap
        self.assertEqual(lt.flags, [])

    def test_outside_window_unknown_with_warning(self):
        manifest = _manifest()
        mic = _whisper([])
        # Segment far from any diar turn → unknown + warning + flag.
        line = _whisper([_seg(0, 1.0, 2.0, "stray utterance")])
        diar = _diar([("SPEAKER_00", 50.0, 60.0)])
        t = merge(manifest, mic, line, diar, generated_at="x")
        lt = [x for x in t.turns if x.channel == "line"][0]
        self.assertEqual(lt.speaker, "unknown")
        self.assertEqual(lt.speaker_id, "unknown")
        self.assertIn("unknown_speaker", lt.flags)
        self.assertTrue(any("unknown" in w for w in t.warnings))


# --- cross-talk overlap flagging ------------------------------------------------------

class TestCrossTalk(unittest.TestCase):
    def test_overlap_flag(self):
        manifest = _manifest()
        mic = _whisper([])
        # One segment overlapping two diar turns near-equally.
        line = _whisper([_seg(0, 10.0, 14.0, "talking over each other")])
        diar = _diar([("SPEAKER_00", 10.0, 12.0), ("SPEAKER_01", 12.0, 14.0)])
        t = merge(manifest, mic, line, diar, generated_at="x")
        lt = [x for x in t.turns if x.channel == "line"][0]
        self.assertIn("overlap", lt.flags)


# --- echo flag + drop -----------------------------------------------------------------

class TestEcho(unittest.TestCase):
    def _echo_inputs(self):
        manifest = _manifest()
        # Mic turn that is a near-copy of a simultaneous line turn = leakage.
        mic = _whisper([_seg(0, 10.0, 13.0, "can we keep it read only during cutover")])
        line = _whisper([_seg(0, 10.1, 13.1, "Can we keep it read only during cutover?")])
        diar = _diar([("SPEAKER_00", 10.0, 14.0)])
        return manifest, mic, line, diar

    def test_echo_flag(self):
        manifest, mic, line, diar = self._echo_inputs()
        t = merge(manifest, mic, line, diar, cfg=MergeConfig(echo_action="flag"),
                  generated_at="x")
        mic_turn = [x for x in t.turns if x.channel == "mic"][0]
        self.assertIn("possible_echo", mic_turn.flags)
        # Both turns survive when flagging.
        self.assertEqual(len([x for x in t.turns if x.channel == "mic"]), 1)

    def test_echo_drop(self):
        manifest, mic, line, diar = self._echo_inputs()
        t = merge(manifest, mic, line, diar, cfg=MergeConfig(echo_action="drop"),
                  generated_at="x")
        self.assertEqual([x for x in t.turns if x.channel == "mic"], [])
        self.assertTrue(any("drop" in w for w in t.warnings))
        # Indices stay contiguous after the drop.
        self.assertEqual([x.index for x in t.turns], list(range(len(t.turns))))

    def test_genuine_simultaneous_speech_not_flagged(self):
        manifest = _manifest()
        # Overlapping in time but different content → kept, no echo flag.
        mic = _whisper([_seg(0, 10.0, 13.0, "let me finish my point first")])
        line = _whisper([_seg(0, 10.1, 13.1, "actually I disagree completely")])
        diar = _diar([("SPEAKER_00", 10.0, 14.0)])
        t = merge(manifest, mic, line, diar, generated_at="x")
        mic_turn = [x for x in t.turns if x.channel == "mic"][0]
        self.assertNotIn("possible_echo", mic_turn.flags)


# --- speakers.json apply + corrections ------------------------------------------------

class TestSpeakersMapApply(unittest.TestCase):
    def test_names_and_corrections(self):
        manifest = _manifest()
        mic = _whisper([_seg(0, 0.0, 2.0, "intro from me")])
        line = _whisper([
            _seg(0, 10.0, 12.0, "first remote line"),
            _seg(1, 20.0, 22.0, "second remote line"),
        ])
        diar = _diar([("SPEAKER_00", 9.5, 13.0), ("SPEAKER_01", 19.5, 23.0)])
        sm = SpeakersMap(
            meeting_id=MID,
            map={"Me": "Paul Nathan", "Speaker_1": "Jane Doe", "Speaker_2": "John Smith"},
            # Reassign the [20,22] turn from Speaker_2 → Speaker_1.
            corrections=[{"start": 19.8, "end": 22.5, "to": "Speaker_1"}],
        )
        t = merge(manifest, mic, line, diar, speakers_map=sm, generated_at="x")

        self.assertEqual(t.turns[0].speaker, "Paul Nathan")
        # First line turn keeps Speaker_1 → Jane Doe.
        first_line = [x for x in t.turns if x.start == 10.0][0]
        self.assertEqual(first_line.speaker, "Jane Doe")
        # Second line turn was reassigned to Speaker_1 (Jane Doe) by the correction.
        second_line = [x for x in t.turns if x.start == 20.0][0]
        self.assertEqual(second_line.speaker_id, "s1")
        self.assertEqual(second_line.speaker, "Jane Doe")
        # Speaker name set on the speakers list too.
        me = [s for s in t.speakers if s.label == "Me"][0]
        self.assertEqual(me.name, "Paul Nathan")

    def test_unmapped_speaker_falls_back_to_label(self):
        manifest = _manifest()
        mic = _whisper([])
        line = _whisper([_seg(0, 10.0, 12.0, "hi")])
        diar = _diar([("SPEAKER_00", 9.5, 13.0)])
        sm = SpeakersMap(meeting_id=MID, map={"Me": "Paul"})  # Speaker_1 unmapped
        t = merge(manifest, mic, line, diar, speakers_map=sm, generated_at="x")
        lt = [x for x in t.turns if x.channel == "line"][0]
        self.assertEqual(lt.speaker, "Speaker_1")


# --- partial meeting propagation ------------------------------------------------------

class TestPartial(unittest.TestCase):
    def test_partial_flag_on_last_turn(self):
        manifest = _manifest(partial=True)
        mic = _whisper([_seg(0, 0.0, 2.0, "start")])
        line = _whisper([_seg(0, 10.0, 12.0, "end")])
        diar = _diar([("SPEAKER_00", 9.5, 13.0)])
        t = merge(manifest, mic, line, diar, generated_at="x")
        self.assertTrue(t.partial)
        self.assertIn("partial", t.turns[-1].flags)
        # Earlier turns are not flagged partial.
        self.assertNotIn("partial", t.turns[0].flags)


# --- edge cases -----------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    def test_empty_diarization_all_speaker_1(self):
        manifest = _manifest()
        mic = _whisper([])
        line = _whisper([
            _seg(0, 10.0, 12.0, "one"),
            _seg(1, 20.0, 22.0, "two"),
        ])
        diar = _diar([])  # silence / no diar
        t = merge(manifest, mic, line, diar, generated_at="x")
        line_turns = [x for x in t.turns if x.channel == "line"]
        # No diar turns at all → everything is unknown (no speaker to assign).
        self.assertTrue(all(x.speaker == "unknown" for x in line_turns))

    def test_single_speaker_all_speaker_1(self):
        manifest = _manifest()
        mic = _whisper([])
        line = _whisper([_seg(0, 10.0, 12.0, "a"), _seg(1, 20.0, 22.0, "b")])
        diar = _diar([("SPEAKER_00", 0.0, 100.0)])
        t = merge(manifest, mic, line, diar, generated_at="x")
        line_turns = [x for x in t.turns if x.channel == "line"]
        self.assertTrue(all(x.speaker == "Speaker_1" for x in line_turns))

    def test_low_confidence_flag(self):
        manifest = _manifest()
        # avg_logprob -1.6 → exp ≈ 0.20 < 0.5 threshold.
        mic = _whisper([_seg(0, 0.0, 2.0, "mumble", avg_logprob=-1.6)])
        line = _whisper([])
        diar = _diar([])
        t = merge(manifest, mic, line, diar, generated_at="x")
        self.assertIn("low_confidence", t.turns[0].flags)


# --- determinism ----------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def test_identical_inputs_identical_output(self):
        manifest = _manifest()
        mic = _whisper([_seg(0, 11.2, 13.0, "alpha")])
        line = _whisper([
            _seg(0, 13.4, 16.1, "beta"),
            _seg(1, 20.0, 22.0, "gamma"),
        ])
        diar = _diar([("SPEAKER_00", 13.0, 17.0), ("SPEAKER_01", 19.5, 23.0)])
        a = merge(manifest, mic, line, diar, generated_at="fixed")
        b = merge(manifest, mic, line, diar, generated_at="fixed")
        self.assertEqual(json.dumps(a.to_dict()), json.dumps(b.to_dict()))

    def test_tie_break_mic_before_line(self):
        manifest = _manifest()
        mic = _whisper([_seg(0, 10.0, 12.0, "mic at ten")])
        line = _whisper([_seg(0, 10.0, 12.0, "line at ten")])
        diar = _diar([("SPEAKER_00", 9.5, 13.0)])
        t = merge(manifest, mic, line, diar, generated_at="x")
        # Same start → mic comes first.
        self.assertEqual(t.turns[0].channel, "mic")
        self.assertEqual(t.turns[1].channel, "line")


# --- disk: validation, outputs, stub --------------------------------------------------

class TestRunDisk(unittest.TestCase):
    def _setup(self, td: Path, with_speakers: bool = False, partial: bool = False):
        rdir = td / "recordings" / MID
        tdir = td / "transcripts" / MID
        rdir.mkdir(parents=True)
        tdir.mkdir(parents=True)
        _manifest(partial=partial).write(rdir / "meeting.json")
        (tdir / "mic.whisper.json").write_text(json.dumps({
            "language": "en", "duration_sec": 30.0,
            "segments": [_seg(0, 1.0, 3.0, "hello from me")],
        }))
        (tdir / "line.whisper.json").write_text(json.dumps({
            "language": "en", "duration_sec": 30.0,
            "segments": [_seg(0, 10.0, 12.0, "remote speaking")],
        }))
        (tdir / "line.diarization.json").write_text(json.dumps({
            "model": "pyannote/x", "duration_sec": 30.0, "num_speakers": 1,
            "segments": [{"speaker": "SPEAKER_00", "start": 9.5, "end": 13.0}],
        }))
        if with_speakers:
            SpeakersMap(MID, {"Me": "Paul", "Speaker_1": "Jane"}).write(tdir / "speakers.json")
        return rdir, tdir

    def test_outputs_and_stub_when_absent(self):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            rdir, tdir = self._setup(td)
            t, warnings = run(MID, tdir, rdir, generated_at="x")
            self.assertTrue((tdir / "transcript.json").exists())
            self.assertTrue((tdir / "transcript.txt").exists())
            # Stub written because speakers.json was absent.
            stub = json.loads((tdir / "speakers.json").read_text())
            self.assertIn("Me", stub["map"])
            self.assertIn("Speaker_1", stub["map"])
            self.assertEqual(stub["map"]["Me"], "")
            self.assertIn("_attendees_hint", stub)
            self.assertTrue(any("stub" in w for w in warnings))
            # transcript.txt is human-readable.
            txt = (tdir / "transcript.txt").read_text()
            self.assertIn("Me: hello from me", txt)

    def test_speakers_applied_no_stub_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            rdir, tdir = self._setup(td, with_speakers=True)
            before = (tdir / "speakers.json").read_text()
            t, _ = run(MID, tdir, rdir, generated_at="x")
            after = (tdir / "speakers.json").read_text()
            self.assertEqual(before, after)  # existing speakers.json untouched
            line_turn = [x for x in t.turns if x.channel == "line"][0]
            self.assertEqual(line_turn.speaker, "Jane")

    def test_missing_required_input_raises_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            rdir, tdir = self._setup(td)
            # Write a sentinel transcript.json that must NOT be touched.
            sentinel = '{"sentinel": true}'
            (tdir / "transcript.json").write_text(sentinel)
            (tdir / "line.whisper.json").unlink()
            with self.assertRaises(InputError):
                run(MID, tdir, rdir, generated_at="x")
            self.assertEqual((tdir / "transcript.json").read_text(), sentinel)

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            rdir, tdir = self._setup(td)
            (tdir / "mic.whisper.json").write_text("{ not json")
            with self.assertRaises(InputError):
                run(MID, tdir, rdir, generated_at="x")

    def test_meeting_id_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            rdir, tdir = self._setup(td)
            with self.assertRaises(InputError):
                run("01J9ZC8Q9F7Y3K2N5R6T8W0XAA", tdir, rdir, generated_at="x")

    def test_rerun_after_editing_speakers_updates_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            rdir, tdir = self._setup(td)
            # First run: no speakers.json → stub + labels.
            run(MID, tdir, rdir, generated_at="x")
            t1 = json.loads((tdir / "transcript.json").read_text())
            line1 = [x for x in t1["turns"] if x["channel"] == "line"][0]
            self.assertEqual(line1["speaker"], "Speaker_1")
            # Human fills the stub.
            SpeakersMap(MID, {"Me": "Paul", "Speaker_1": "Jane"}).write(tdir / "speakers.json")
            # Re-run applies it.
            run(MID, tdir, rdir, generated_at="x")
            t2 = json.loads((tdir / "transcript.json").read_text())
            line2 = [x for x in t2["turns"] if x["channel"] == "line"][0]
            self.assertEqual(line2["speaker"], "Jane")

    def test_main_returns_zero_and_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            rdir, tdir = self._setup(td)
            rc = main(["--meeting-id", MID, "--transcripts-dir", str(tdir),
                       "--recordings-dir", str(rdir)])
            self.assertEqual(rc, 0)
            # Now break a required input → non-zero exit (no exception).
            (tdir / "line.diarization.json").unlink()
            rc = main(["--meeting-id", MID, "--transcripts-dir", str(tdir),
                       "--recordings-dir", str(rdir)])
            self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
