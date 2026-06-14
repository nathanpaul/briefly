"""Preprocess stage tests — stdlib only, no hardware, no 3rd-party libs.

Synthetic WAVs are generated with stdlib `wave`; the tests shell out to ffmpeg (installed)
for the actual de-clip/normalize/resample. They must run WITHOUT the optional AEC backend:
the AEC-on test asserts graceful passthrough fallback (aec_enabled=false in the report),
not a failure. A tiny embedded JSON-Schema checker validates preprocess.json against
briefly/schemas/preprocess.schema.json without pulling in `jsonschema`.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from briefly.audio import preprocess as pp
from briefly.audio.preprocess import PreprocessConfig, preprocess
from briefly.models import CaptureInfo, ChannelInfo, MeetingManifest

FFMPEG = "/opt/homebrew/bin/ffmpeg"
HAVE_FFMPEG = shutil.which(FFMPEG) is not None or os.path.exists(FFMPEG)
MEETING_ID = "01J9ZC8Q9F7Y3K2N5R6T8W0X1Z"  # valid 26-char Crockford ULID


# --------------------------------------------------------------------------- #
# Synthetic WAV helpers (stdlib wave).
# --------------------------------------------------------------------------- #
def _write_wav(path: Path, samples: list[int], rate: int = 48000) -> None:
    clamped = [max(-32768, min(32767, int(s))) for s in samples]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<%dh" % len(clamped), *clamped))


def _sine(freq: float, dur_sec: float, rate: int = 48000, amp: int = 12000,
          phase: float = 0.0) -> list[int]:
    n = int(dur_sec * rate)
    return [int(amp * math.sin(2 * math.pi * freq * i / rate + phase)) for i in range(n)]


def _read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        nch = w.getnchannels()
        width = w.getsampwidth()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
    return rate, nch, width, nframes, raw


def _make_manifest(d: Path, line_offset: float = 0.0) -> None:
    m = MeetingManifest(
        meeting_id=MEETING_ID,
        date="2026-06-14",
        started_at="2026-06-14T09:00:00Z",
        ended_at="2026-06-14T09:00:02Z",
        partial=False,
        attendees=[],
        capture=CaptureInfo("dual-process", 48000, "pcm_s16le", 1, "8.1.1",
                            "process-start-delta"),
        channels={
            "mic": ChannelInfo("mic.wav", "Cubilux CB5 MIC2", 0.0, speaker="Me"),
            "line": ChannelInfo("line.wav", "Cubilux CB5 Line In", line_offset),
        },
    )
    m.write(d / "meeting.json")


# --------------------------------------------------------------------------- #
# Minimal JSON-Schema validator (subset: type/const/enum/required/properties/
# additionalProperties/$ref/$defs/pattern). Enough for preprocess.schema.json.
# --------------------------------------------------------------------------- #
def _validate(instance, schema, root=None, path="$"):
    import re
    root = root or schema
    errs: list[str] = []

    def resolve(s):
        if "$ref" in s:
            ref = s["$ref"]
            assert ref.startswith("#/")
            node = root
            for part in ref[2:].split("/"):
                node = node[part]
            return node
        return s

    schema = resolve(schema)

    if "const" in schema and instance != schema["const"]:
        errs.append(f"{path}: {instance!r} != const {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errs.append(f"{path}: {instance!r} not in enum {schema['enum']}")
    if "type" in schema:
        types = schema["type"]
        types = [types] if isinstance(types, str) else types
        ok = False
        for t in types:
            if t == "object" and isinstance(instance, dict):
                ok = True
            elif t == "array" and isinstance(instance, list):
                ok = True
            elif t == "string" and isinstance(instance, str):
                ok = True
            elif t == "integer" and isinstance(instance, int) and not isinstance(instance, bool):
                ok = True
            elif t == "number" and isinstance(instance, (int, float)) and not isinstance(instance, bool):
                ok = True
            elif t == "boolean" and isinstance(instance, bool):
                ok = True
            elif t == "null" and instance is None:
                ok = True
        if not ok:
            errs.append(f"{path}: {instance!r} not of type {types}")
    if "pattern" in schema and isinstance(instance, str):
        if not re.search(schema["pattern"], instance):
            errs.append(f"{path}: {instance!r} fails pattern {schema['pattern']}")
    if isinstance(instance, dict):
        for req in schema.get("required", []):
            if req not in instance:
                errs.append(f"{path}: missing required {req!r}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for k in instance:
                if k not in props:
                    errs.append(f"{path}: unexpected property {k!r}")
        for k, v in instance.items():
            if k in props:
                errs += _validate(v, props[k], root, f"{path}.{k}")
    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errs += _validate(item, schema["items"], root, f"{path}[{i}]")
    return errs


def _load_schema() -> dict:
    schema_path = Path(pp.__file__).parent.parent / "schemas" / "preprocess.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.rec = Path(self.tmp) / "recordings" / MEETING_ID
        self.rec.mkdir(parents=True)
        self.proc = Path(self.tmp) / "processed"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _basic_inputs(self, line_offset: float = 0.0):
        _write_wav(self.rec / "mic.wav", _sine(220, 0.4))      # near-end "voice"
        _write_wav(self.rec / "line.wav", _sine(440, 0.4))     # remote reference
        _make_manifest(self.rec, line_offset=line_offset)

    # ---- core: resample to 16 kHz mono + schema-valid report ----------------
    def test_outputs_are_16k_mono_and_schema_valid(self):
        self._basic_inputs()
        res = preprocess(MEETING_ID, self.rec.parent, self.proc,
                         PreprocessConfig(aec_enabled=False))
        outdir = self.proc / MEETING_ID
        for name in ("mic.16k.wav", "line.16k.wav", "preprocess.json"):
            self.assertTrue((outdir / name).exists(), name)
        for name in ("mic.16k.wav", "line.16k.wav"):
            rate, nch, width, nframes, _ = _read_wav(outdir / name)
            self.assertEqual(rate, 16000, f"{name} sample rate")
            self.assertEqual(nch, 1, f"{name} channels")
            self.assertEqual(width, 2, f"{name} sample width")
            self.assertGreater(nframes, 0)

        report = json.loads((outdir / "preprocess.json").read_text())
        errs = _validate(report, _load_schema())
        self.assertEqual(errs, [], "schema errors:\n" + "\n".join(errs))
        self.assertEqual(report["meeting_id"], MEETING_ID)
        self.assertEqual(report["resample_rate"], 16000)
        # report returned in-memory matches what was written
        self.assertEqual(res.report["meeting_id"], MEETING_ID)

    # ---- AEC off ⇒ mic preserved (passthrough; no AEC) ----------------------
    def test_aec_off_passthrough_preserves_mic(self):
        self._basic_inputs()
        res = preprocess(MEETING_ID, self.rec.parent, self.proc,
                         PreprocessConfig(aec_enabled=False))
        self.assertFalse(res.report["aec_enabled"])
        self.assertFalse(res.report["channels"]["mic"]["aec_applied"])
        self.assertIsNone(res.report["estimated_echo_reduction_db"])
        self.assertEqual(res.report["params"]["aec_enabled"], False)
        # the 16k mic still carries the 220 Hz tone (energy preserved, not cancelled)
        self.assertIsNotNone(res.report["channels"]["mic"]["after"]["mean_dbfs"])
        self.assertGreater(res.report["channels"]["mic"]["after"]["mean_dbfs"], -60.0)

    # ---- AEC on, backend absent ⇒ graceful fallback (NOT a failure) ---------
    def test_aec_on_without_backend_falls_back(self):
        self._basic_inputs(line_offset=0.0)
        available, _ = pp._aec_backend_available()
        res = preprocess(MEETING_ID, self.rec.parent, self.proc,
                         PreprocessConfig(aec_enabled=True))
        if available:
            self.skipTest("an AEC backend is installed; fallback path not exercised")
        # No backend: report records aec_enabled=false + a clear reason, still exits clean.
        self.assertFalse(res.report["aec_enabled"])
        self.assertFalse(res.report["channels"]["mic"]["aec_applied"])
        self.assertTrue(any("backend" in w.lower() for w in res.report["warnings"]),
                        res.report["warnings"])
        # params record what was REQUESTED (true), report records what HAPPENED (false).
        self.assertTrue(res.report["params"]["aec_enabled"])
        errs = _validate(res.report, _load_schema())
        self.assertEqual(errs, [])

    # ---- clipping detection on a deliberately hot WAV -----------------------
    def test_clipping_detected_on_hot_wav(self):
        # mic full-scale (clipped), line clean
        hot = [32767 if (i // 50) % 2 == 0 else -32768 for i in range(19200)]
        _write_wav(self.rec / "mic.wav", hot)
        _write_wav(self.rec / "line.wav", _sine(440, 0.4))
        _make_manifest(self.rec)
        res = preprocess(MEETING_ID, self.rec.parent, self.proc,
                         PreprocessConfig(aec_enabled=False))
        self.assertTrue(res.report["channels"]["mic"]["before"]["clipping_detected"])
        self.assertFalse(res.report["channels"]["line"]["before"]["clipping_detected"])
        self.assertTrue(any("clip" in w.lower() for w in res.report["warnings"]),
                        res.report["warnings"])
        # warn-but-proceed: valid 16k outputs still produced
        outdir = self.proc / MEETING_ID
        rate, nch, _, nframes, _ = _read_wav(outdir / "mic.16k.wav")
        self.assertEqual((rate, nch), (16000, 1))
        self.assertGreater(nframes, 0)

    def test_detect_clipping_helper(self):
        with tempfile.TemporaryDirectory() as td:
            hot = Path(td) / "hot.wav"
            quiet = Path(td) / "quiet.wav"
            _write_wav(hot, [32767] * 1000)
            _write_wav(quiet, _sine(300, 0.2, amp=4000))
            self.assertTrue(pp.detect_clipping(hot, -0.1))
            self.assertFalse(pp.detect_clipping(quiet, -0.1))

    # ---- line silent ⇒ AEC skipped + warning, still normalized/resampled ----
    def test_line_silent_skips_aec(self):
        _write_wav(self.rec / "mic.wav", _sine(220, 0.4))
        _write_wav(self.rec / "line.wav", [0] * 19200)  # digital silence
        _make_manifest(self.rec)
        res = preprocess(MEETING_ID, self.rec.parent, self.proc,
                         PreprocessConfig(aec_enabled=True))
        self.assertFalse(res.report["aec_enabled"])
        self.assertTrue(any("silent" in w.lower() for w in res.report["warnings"]),
                        res.report["warnings"])
        outdir = self.proc / MEETING_ID
        for name in ("mic.16k.wav", "line.16k.wav"):
            rate, nch, _, _, _ = _read_wav(outdir / name)
            self.assertEqual((rate, nch), (16000, 1))

    # ---- delay alignment: a known offset is recovered by xcorr --------------
    def test_xcorr_recovers_injected_delay(self):
        rate = 48000
        # Broadband (noise-like) reference: a pure sine is periodic and gives ambiguous
        # correlation peaks every period; real far-end audio (speech) is broadband, so a
        # deterministic LCG-noise reference is the realistic, well-posed test fixture.
        seed = 1234567
        ref = []
        for _ in range(rate):  # 1.0 s
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            ref.append(((seed >> 8) % 20000) - 10000)
        lag = 480  # 10 ms at 48 kHz
        # mic = near-end (uncorrelated tone) + delayed copy of the reference (the echo)
        near = _sine(180, 1.0, rate=rate, amp=6000)
        echo = [0] * lag + ref[: len(ref) - lag]
        mic = [near[i] + echo[i] for i in range(len(ref))]
        _write_wav(self.rec / "mic.wav", mic, rate=rate)
        _write_wav(self.rec / "line.wav", ref, rate=rate)
        _make_manifest(self.rec, line_offset=0.0)
        refined = pp.refine_delay_xcorr(self.rec / "mic.wav", self.rec / "line.wav",
                                        coarse_sec=0.0, search_ms=30.0)
        self.assertIsNotNone(refined)
        # within ~2 ms of the true 10 ms injected delay (decimation-bounded)
        self.assertAlmostEqual(refined, lag / rate, delta=0.002)

        res = preprocess(MEETING_ID, self.rec.parent, self.proc,
                         PreprocessConfig(aec_enabled=False, xcorr_refine=True))
        self.assertEqual(res.report["delay_source"], "meeting.json+xcorr")
        self.assertAlmostEqual(res.report["delay_applied_sec"], lag / rate, delta=0.002)

    def test_coarse_delay_from_manifest(self):
        _make_manifest(self.rec, line_offset=0.021)
        m = MeetingManifest.read(self.rec / "meeting.json")
        self.assertAlmostEqual(pp.coarse_delay_sec(m), 0.021, places=6)

    # ---- missing input ⇒ correct non-zero exit, no output written -----------
    def test_missing_mic_input_nonzero_exit(self):
        # only line + manifest present
        _write_wav(self.rec / "line.wav", _sine(440, 0.5))
        _make_manifest(self.rec)
        with self.assertRaises(pp.InputError) as cm:
            preprocess(MEETING_ID, self.rec.parent, self.proc, PreprocessConfig())
        self.assertEqual(cm.exception.exit_code, 2)

    def test_missing_manifest_nonzero_exit(self):
        _write_wav(self.rec / "mic.wav", _sine(220, 0.5))
        _write_wav(self.rec / "line.wav", _sine(440, 0.5))
        with self.assertRaises(pp.InputError):
            preprocess(MEETING_ID, self.rec.parent, self.proc, PreprocessConfig())

    def test_cli_missing_input_returns_nonzero(self):
        _write_wav(self.rec / "line.wav", _sine(440, 0.5))
        _make_manifest(self.rec)
        rc = pp.main(["--meeting-id", MEETING_ID,
                      "--recordings-dir", str(self.rec.parent),
                      "--processed-dir", str(self.proc)])
        self.assertEqual(rc, 2)

    # ---- CLI happy path + --no-aec ------------------------------------------
    def test_cli_no_aec_success(self):
        self._basic_inputs()
        rc = pp.main(["--meeting-id", MEETING_ID,
                      "--recordings-dir", str(self.rec.parent),
                      "--processed-dir", str(self.proc),
                      "--no-aec"])
        self.assertEqual(rc, 0)
        report = json.loads((self.proc / MEETING_ID / "preprocess.json").read_text())
        self.assertFalse(report["aec_enabled"])

    # ---- idempotency: two runs ⇒ byte-identical audio outputs ----------------
    def test_idempotent_outputs(self):
        self._basic_inputs()
        cfg = PreprocessConfig(aec_enabled=False)
        preprocess(MEETING_ID, self.rec.parent, self.proc, cfg)
        first = {n: (self.proc / MEETING_ID / n).read_bytes()
                 for n in ("mic.16k.wav", "line.16k.wav")}
        preprocess(MEETING_ID, self.rec.parent, self.proc, cfg)
        for n, b in first.items():
            self.assertEqual((self.proc / MEETING_ID / n).read_bytes(), b,
                             f"{n} not byte-identical across runs")

    # ---- accepts recordings/<id>/ passed directly OR the parent root ---------
    def test_accepts_direct_meeting_dir(self):
        self._basic_inputs()
        res = preprocess(MEETING_ID, self.rec, self.proc,
                         PreprocessConfig(aec_enabled=False))
        self.assertTrue((res.processed_dir / "preprocess.json").exists())


class TestConfigParsing(unittest.TestCase):
    """No ffmpeg/hardware needed — pure config parsing."""

    def test_from_json_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.json"
            p.write_text(json.dumps({"aec_enabled": False, "resample_rate": 8000}))
            cfg = PreprocessConfig.from_file(p)
            self.assertFalse(cfg.aec_enabled)
            self.assertEqual(cfg.resample_rate, 8000)
            self.assertEqual(cfg.normalize_target_dbfs, -3.0)  # default preserved

    def test_from_yaml_subset(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.yaml"
            p.write_text("aec_enabled: false\nnormalize_target_dbfs: -6.0\n# comment\n")
            cfg = PreprocessConfig.from_file(p)
            self.assertFalse(cfg.aec_enabled)
            self.assertEqual(cfg.normalize_target_dbfs, -6.0)

    def test_defaults(self):
        cfg = PreprocessConfig()
        self.assertTrue(cfg.aec_enabled)
        self.assertEqual(cfg.normalize_target_dbfs, -3.0)
        self.assertEqual(cfg.resample_rate, 16000)
        self.assertTrue(cfg.declip)
        self.assertEqual(cfg.ffmpeg_path, "/opt/homebrew/bin/ffmpeg")


if __name__ == "__main__":
    unittest.main()
