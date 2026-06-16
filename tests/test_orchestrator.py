import json
import tempfile
import unittest
from pathlib import Path

from briefly.models import CaptureInfo, ChannelInfo, MeetingManifest, Transcript
from briefly.orchestrator import PipelineConfig, notes_path, run_pipeline

MID = "01J9ZC8Q9F7Y3K2N5R6T8W0X1Z"
_QUIET = lambda *a, **k: None  # noqa: E731

MIC_WHISPER = {"language": "en", "duration_sec": 5.0,
               "segments": [{"id": 0, "start": 0.2, "end": 2.0, "text": "Hello from me",
                             "avg_logprob": -0.2, "no_speech_prob": 0.01}]}
LINE_WHISPER = {"language": "en", "duration_sec": 5.0,
                "segments": [{"id": 0, "start": 2.2, "end": 4.0, "text": "Hello from the client",
                              "avg_logprob": -0.2, "no_speech_prob": 0.01}]}
LINE_DIAR = {"model": "pyannote/community-1", "duration_sec": 5.0, "num_speakers": 1,
             "segments": [{"speaker": "SPEAKER_00", "start": 2.1, "end": 4.1}]}


def _setup_meeting(root: Path) -> None:
    rec = root / "recordings" / MID
    rec.mkdir(parents=True)
    (rec / "mic.wav").write_bytes(b"x")
    (rec / "line.wav").write_bytes(b"x")
    MeetingManifest(
        meeting_id=MID, date="2026-06-14", started_at="2026-06-14T09:00:00Z",
        ended_at="2026-06-14T09:05:00Z", partial=False, attendees=["Jane Doe"],
        capture=CaptureInfo("dual-process", 48000, "pcm_s16le", 2, "8.1.1", "process-start-delta"),
        channels={
            "mic": ChannelInfo("mic.wav", "Cubilux CB5 MIC2", 0.0, speaker="Me",
                               duration_sec=5.0, peak_dbfs=-6.0, mean_dbfs=-20.0, clipping=False),
            "line": ChannelInfo("line.wav", "Cubilux CB5 Line In", 0.0,
                                duration_sec=5.0, peak_dbfs=-6.0, mean_dbfs=-20.0, clipping=False),
        },
    ).write(rec / "meeting.json")


def _fakes(calls: list):
    def fake_preprocess(cfg, mid, progress=None):
        calls.append("preprocess")
        cfg.proc(mid).mkdir(parents=True, exist_ok=True)
        (cfg.proc(mid) / "mic.16k.wav").write_bytes(b"x")
        (cfg.proc(mid) / "line.16k.wav").write_bytes(b"x")

    def fake_transcribe(cfg, mid, progress=None):
        calls.append("transcribe")
        cfg.tx(mid).mkdir(parents=True, exist_ok=True)
        (cfg.tx(mid) / "mic.whisper.json").write_text(json.dumps(MIC_WHISPER))
        (cfg.tx(mid) / "line.whisper.json").write_text(json.dumps(LINE_WHISPER))

    def fake_diarize(cfg, mid, progress=None):
        calls.append("diarize")
        cfg.tx(mid).mkdir(parents=True, exist_ok=True)   # diarize now runs before transcribe
        (cfg.tx(mid) / "line.diarization.json").write_text(json.dumps(LINE_DIAR))

    def fake_summarize(cfg, mid, progress=None):
        calls.append("summarize")
        p = notes_path(cfg, mid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("---\ntype: meeting\n---\n# Note\n"
                     "<!-- briefly:enrichment:start -->\n<!-- briefly:enrichment:end -->\n")

    def fake_enrich(cfg, mid, progress=None):
        calls.append("enrich")

    return {"preprocess": fake_preprocess, "transcribe": fake_transcribe,
            "diarize": fake_diarize, "summarize": fake_summarize, "enrich": fake_enrich}


class TestOrchestrator(unittest.TestCase):
    def test_end_to_end_real_merge(self):
        calls: list = []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _setup_meeting(root)
            cfg = PipelineConfig(data_root=str(root), vault_dir=str(root / "vault"))
            results = run_pipeline(cfg, MID, "preprocess", "enrich",
                                   runners=_fakes(calls), log=_QUIET)
            self.assertEqual([s for s, _ in results], ["preprocess", "diarize", "transcribe",
                                                       "merge", "summarize", "enrich"])
            self.assertTrue(all(state == "ok" for _, state in results))
            # real merge produced a valid transcript with a "Me" turn and a Speaker_1 turn
            t = Transcript.read(cfg.tx(MID) / "transcript.json")
            self.assertTrue(any(turn.channel == "mic" for turn in t.turns))
            self.assertTrue(any(s.label == "Speaker_1" for s in t.speakers))
            self.assertTrue(notes_path(cfg, MID).exists())

    def test_idempotent_skip_on_rerun(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _setup_meeting(root)
            cfg = PipelineConfig(data_root=str(root), vault_dir=str(root / "vault"))
            run_pipeline(cfg, MID, "preprocess", "merge", runners=_fakes([]), log=_QUIET)
            calls: list = []
            results = run_pipeline(cfg, MID, "preprocess", "merge",
                                   runners=_fakes(calls), log=_QUIET)
            self.assertEqual(calls, [])  # nothing re-run
            self.assertTrue(all(state == "skip" for _, state in results))

    def test_from_to_slice(self):
        calls: list = []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _setup_meeting(root)
            cfg = PipelineConfig(data_root=str(root), vault_dir=str(root / "vault"))
            # produce the merge inputs only (preprocess..transcribe covers diarize too)
            run_pipeline(cfg, MID, "preprocess", "transcribe", runners=_fakes(calls), log=_QUIET)
            calls.clear()
            results = run_pipeline(cfg, MID, "merge", "merge", runners=_fakes(calls), log=_QUIET)
            self.assertEqual([s for s, _ in results], ["merge"])
            self.assertEqual(calls, [])  # only real merge ran; no fake stage invoked

    def test_force_reruns(self):
        calls: list = []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _setup_meeting(root)
            cfg = PipelineConfig(data_root=str(root), vault_dir=str(root / "vault"))
            run_pipeline(cfg, MID, "preprocess", "transcribe", runners=_fakes(calls), log=_QUIET)
            calls.clear()
            run_pipeline(cfg, MID, "preprocess", "transcribe", force=True,
                         runners=_fakes(calls), log=_QUIET)
            self.assertEqual(calls, ["preprocess", "diarize", "transcribe"])


if __name__ == "__main__":
    unittest.main()
