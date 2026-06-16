"""Tests for the summarize stage. Stdlib unittest only; NO network — a fake Claude
client is injected. Covers the golden render, enrich-block preservation on re-run,
unmapped Speaker_N handling, partial propagation, back-fill, empty transcript, and the
end-to-end fake-client path."""
import json
import tempfile
import unittest
from pathlib import Path

from briefly.models import (CaptureInfo, ChannelInfo, MeetingManifest, Speaker,
                            SpeakersMap, Transcript, Turn)
from briefly import summarize as S


# --------------------------------------------------------------------------- fixtures


def _transcript(partial: bool = False, with_speaker2: bool = True,
                me_named: bool = True) -> Transcript:
    speakers = [
        Speaker(id="me", label="Me", channel="mic", source="channel",
                name="Paul Nathan" if me_named else None),
        Speaker(id="s1", label="Speaker_1", channel="line", source="diarization", name="Jane Doe"),
    ]
    turns = [
        Turn(0, "me", "Paul Nathan" if me_named else "Me", "mic", 11.2, 13.0,
             "Let me walk through the migration phasing."),
        Turn(1, "s1", "Jane Doe", "line", 13.4, 16.1,
             "Can we keep the legacy warehouse read-only during the cutover?"),
    ]
    if with_speaker2:
        speakers.append(Speaker(id="s2", label="Speaker_2", channel="line",
                                source="diarization"))
        turns.append(Turn(2, "s2", "Speaker_2", "line", 16.5, 18.0,
                          "Which ETL tool moves the warehouse tables?"))
    return Transcript(
        meeting_id="01J9ZC8Q9F7Y3K2N5R6T8W0X1Z",
        date="2026-06-14",
        generated_at="2026-06-14T10:05:00Z",
        partial=partial,
        models={"transcription": "whisper-large-v3"},
        speakers=speakers,
        turns=turns,
    )


def _manifest() -> MeetingManifest:
    return MeetingManifest(
        meeting_id="01J9ZC8Q9F7Y3K2N5R6T8W0X1Z",
        date="2026-06-14",
        started_at="2026-06-14T10:00:00Z",
        ended_at="2026-06-14T10:30:00Z",
        partial=False,
        attendees=["[[Jane Doe]]"],
        capture=CaptureInfo(mode="dual-process", sample_rate=48000, format="pcm_s16le",
                            channels=2, ffmpeg="7.0", offset_method="process-start-delta"),
        channels={
            "mic": ChannelInfo(file="mic.wav", device_name="Cubilux CB5 MIC2",
                               start_offset_sec=0.0, speaker="Me"),
            "line": ChannelInfo(file="line.wav", device_name="Cubilux CB5 Line In",
                                start_offset_sec=0.1),
        },
    )


def _full_brief() -> dict:
    return {
        "headline": "Kickoff on the Acme AWS migration — landing-zone approach, phasing, and downtime concerns.",
        "per_speaker": [
            {"speaker": "Paul Nathan",
             "summary": ["Walked through the proposed landing-zone approach and migration phasing."],
             "questions": []},
            {"speaker": "Jane Doe",
             "summary": ["Wants the migration to avoid downtime for the billing service."],
             "questions": ["What's the rollback plan if the cutover fails?",
                           "Can we keep the legacy data warehouse read-only during transition?"]},
            {"speaker": "Speaker_2",
             "summary": ["Asked about the data-migration tooling but was not named in speakers.json."],
             "questions": ["Which ETL tool moves the warehouse tables?"]},
        ],
        "open_questions": [
            {"question": "Rollback plan for cutover", "owner": "us"},
            {"question": "Read-only legacy warehouse during transition — feasible?", "owner": "us"},
            {"question": "ETL tool for warehouse tables", "owner": "us"},
        ],
    }


def _write_inputs(root: Path, transcript: Transcript, manifest: MeetingManifest,
                  speakers_map: SpeakersMap | None = None) -> S.SummarizeConfig:
    tdir = root / "transcripts" / transcript.meeting_id
    rdir = root / "recordings" / transcript.meeting_id
    tdir.mkdir(parents=True)
    rdir.mkdir(parents=True)
    transcript.write(tdir / "transcript.json")
    manifest.write(rdir / "meeting.json")
    if speakers_map is not None:
        speakers_map.write(tdir / "speakers.json")
    return S.SummarizeConfig(
        transcripts_dir=str(root / "transcripts"),
        recordings_dir=str(root / "recordings"),
        vault_dir=str(root / "vault"),
        project="[[Apollo MOC]]",
        proposal="[[Apollo-Platform-Migration]]",
    )


def _fake_client(brief: dict):
    """Return a BriefClient that records calls and returns the given brief."""
    calls = []

    def _call(*, system, transcript_text, schema, model, max_tokens, max_retries):
        calls.append({"system": system, "transcript_text": transcript_text,
                      "model": model})
        return json.loads(json.dumps(brief))  # deep copy so the renderer can't mutate it

    _call.calls = calls
    return _call


# ---------------------------------------------------------------------- golden render


class TestGoldenRender(unittest.TestCase):
    def setUp(self):
        self.cfg = S.SummarizeConfig(project="[[Apollo MOC]]", proposal="[[Apollo-Platform-Migration]]")
        self.transcript = _transcript()
        self.manifest = _manifest()
        self.note = S.render_notes(S.reconcile_brief(_full_brief(), self.transcript),
                                   self.transcript, self.manifest, self.cfg)

    def test_frontmatter(self):
        fm = self.note.split("---", 2)[1]
        self.assertIn("type: meeting", fm)
        self.assertIn("meeting_id: 01J9ZC8Q9F7Y3K2N5R6T8W0X1Z", fm)
        self.assertIn("date: 2026-06-14", fm)
        self.assertIn('project: "[[Apollo MOC]]"', fm)
        self.assertIn('attendees: ["[[Jane Doe]]"]', fm)
        self.assertIn('proposal: "[[Apollo-Platform-Migration]]"', fm)
        self.assertIn("status: draft", fm)
        self.assertIn("partial: false", fm)
        self.assertIn("tags: [meeting]", fm)
        self.assertIn("summary:", fm)

    def test_title_from_headline(self):
        self.assertIn("# Kickoff on the Acme AWS migration", self.note)

    def test_me_heading_named(self):
        # Me speaker renders as "## Me (Paul Nathan)", not "## Paul Nathan".
        self.assertIn("## Me (Paul Nathan)", self.note)
        self.assertNotIn("## Paul Nathan", self.note)

    def test_per_speaker_sections(self):
        self.assertIn("## Jane Doe", self.note)
        self.assertIn("- Wants the migration to avoid downtime", self.note)
        # Me has no questions → "none." line.
        me_block = self.note.split("## Me (Paul Nathan)", 1)[1].split("##", 1)[0]
        self.assertIn("- Questions raised: none.", me_block)
        # Jane has questions → nested list.
        jane_block = self.note.split("## Jane Doe", 1)[1].split("##", 1)[0]
        self.assertIn("- Questions raised:", jane_block)
        self.assertIn("  - What's the rollback plan if the cutover fails?", jane_block)

    def test_open_questions(self):
        oq = self.note.split("## Open Questions", 1)[1]
        self.assertIn("- Rollback plan for cutover (owner: us).", oq)
        self.assertIn("- ETL tool for warehouse tables (owner: us).", oq)

    def test_empty_managed_block_on_first_render(self):
        self.assertIn(S.ENRICH_PLACEHOLDER, self.note)
        # Ordering: Open Questions precedes the managed block.
        self.assertLess(self.note.index("## Open Questions"), self.note.index(S.ENRICH_START))

    def test_speaker_order_follows_transcript(self):
        # Me, then Jane, then Speaker_2.
        i_me = self.note.index("## Me (Paul Nathan)")
        i_jane = self.note.index("## Jane Doe")
        i_s2 = self.note.index("## Speaker_2")
        self.assertLess(i_me, i_jane)
        self.assertLess(i_jane, i_s2)


# ------------------------------------------------------------- enrich-block preservation


class TestEnrichBlockPreservation(unittest.TestCase):
    def test_extract_block(self):
        filled = (f"{S.ENRICH_START}\n## Context\n- Client: [[Acme MOC]]\n{S.ENRICH_END}")
        text = f"# Note\n\n## Open Questions\n\n{filled}\n"
        self.assertEqual(S.extract_enrich_block(text), filled)

    def test_extract_block_absent(self):
        self.assertIsNone(S.extract_enrich_block("# Note\nno block here"))

    def test_rerun_preserves_filled_block_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transcript = _transcript()
            cfg = _write_inputs(root, transcript, _manifest())
            out = S.output_path(cfg, transcript)

            # First run: write a note, then hand-edit a FILLED enrich block into it.
            S.summarize(cfg, transcript.meeting_id, client=_fake_client(_full_brief()))
            filled = (
                f"{S.ENRICH_START}\n"
                "## Context\n"
                "- Client: [[Acme MOC]] · Proposal: [[Acme-AWS-Migration]]\n"
                "- Attendees: [[Jane Doe]] (CTO)\n\n"
                "## Connections & follow-ups\n"
                "- Links to [[Prior Acme Engagement]]\n"
                f"{S.ENRICH_END}"
            )
            original = out.read_text(encoding="utf-8")
            placeholder = S.extract_enrich_block(original)
            seeded = original.replace(placeholder, filled)
            out.write_text(seeded, encoding="utf-8")

            # Re-run with a CHANGED brief (sections must update, block must survive).
            changed = _full_brief()
            changed["per_speaker"][1]["summary"] = ["UPDATED: insists on zero downtime."]
            S.summarize(cfg, transcript.meeting_id, client=_fake_client(changed))

            after = out.read_text(encoding="utf-8")
            # Enrich block preserved byte-for-byte.
            self.assertEqual(S.extract_enrich_block(after), filled)
            self.assertIn("## Connections & follow-ups", after)
            self.assertIn("[[Prior Acme Engagement]]", after)
            # Sections regenerated.
            self.assertIn("UPDATED: insists on zero downtime.", after)
            # No placeholder text leaked back in, no duplicate block.
            self.assertEqual(after.count(S.ENRICH_START), 1)
            self.assertEqual(after.count(S.ENRICH_END), 1)

    def test_rerun_with_placeholder_stays_placeholder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transcript = _transcript()
            cfg = _write_inputs(root, transcript, _manifest())
            out = S.output_path(cfg, transcript)
            S.summarize(cfg, transcript.meeting_id, client=_fake_client(_full_brief()))
            S.summarize(cfg, transcript.meeting_id, client=_fake_client(_full_brief()))
            text = out.read_text(encoding="utf-8")
            self.assertIn(S.ENRICH_PLACEHOLDER, text)
            self.assertEqual(text.count(S.ENRICH_START), 1)


# ---------------------------------------------------------------- speaker reconciliation


class TestReconcile(unittest.TestCase):
    def test_backfill_missing_speaker(self):
        # Claude omits Speaker_2 → app back-fills an empty section.
        brief = _full_brief()
        brief["per_speaker"] = [ps for ps in brief["per_speaker"] if ps["speaker"] != "Speaker_2"]
        out = S.reconcile_brief(brief, _transcript())
        names = [ps["speaker"] for ps in out["per_speaker"]]
        self.assertEqual(names, ["Paul Nathan", "Jane Doe", "Speaker_2"])
        s2 = next(ps for ps in out["per_speaker"] if ps["speaker"] == "Speaker_2")
        self.assertEqual(s2["summary"], [])
        self.assertEqual(s2["questions"], [])

    def test_drop_unknown_speaker(self):
        brief = _full_brief()
        brief["per_speaker"].append({"speaker": "Ghost", "summary": ["x"], "questions": []})
        out = S.reconcile_brief(brief, _transcript())
        self.assertNotIn("Ghost", [ps["speaker"] for ps in out["per_speaker"]])

    def test_unmapped_speaker_section_rendered(self):
        cfg = S.SummarizeConfig()
        transcript = _transcript()
        note = S.render_notes(S.reconcile_brief(_full_brief(), transcript),
                              transcript, _manifest(), cfg)
        self.assertIn("## Speaker_2", note)
        self.assertIn("- Which ETL tool moves the warehouse tables?", note)


# ------------------------------------------------------------------- partial / fallback


class TestEdgeCases(unittest.TestCase):
    def test_partial_propagates_to_frontmatter_and_prompt(self):
        transcript = _transcript(partial=True)
        cfg = S.SummarizeConfig()
        note = S.render_notes(S.reconcile_brief(_full_brief(), transcript),
                              transcript, _manifest(), cfg)
        self.assertIn("partial: true", note)
        system, _ = S.build_brief_prompt(transcript)
        self.assertIn("PARTIAL", system)

    def test_missing_speakers_json_uses_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transcript = _transcript()  # Speaker_2 unnamed, no speakers.json
            cfg = _write_inputs(root, transcript, _manifest())
            out = S.summarize(cfg, transcript.meeting_id, client=_fake_client(_full_brief()))
            self.assertIn("## Speaker_2", out.read_text(encoding="utf-8"))

    def test_speakers_json_fallback_name(self):
        # transcript lacks a resolved name on Speaker_2; speakers.json supplies it.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transcript = _transcript()
            sm = SpeakersMap(transcript.meeting_id, {"Speaker_2": "Bob Smith"})
            cfg = _write_inputs(root, transcript, _manifest(), speakers_map=sm)
            names = S.speaker_display_names(transcript, sm)
            self.assertIn("Bob Smith", names)
            brief = _full_brief()
            brief["per_speaker"][2]["speaker"] = "Bob Smith"
            out = S.summarize(cfg, transcript.meeting_id, client=_fake_client(brief))
            self.assertIn("## Bob Smith", out.read_text(encoding="utf-8"))

    def test_me_unnamed_renders_bare(self):
        transcript = _transcript(me_named=False)
        cfg = S.SummarizeConfig()
        brief = _full_brief()
        brief["per_speaker"][0]["speaker"] = "Me"
        note = S.render_notes(S.reconcile_brief(brief, transcript),
                              transcript, _manifest(), cfg)
        self.assertIn("## Me", note)
        self.assertNotIn("## Me (", note)

    def test_empty_transcript_writes_minimal_note_no_client_call(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transcript = _transcript()
            transcript.turns = []
            cfg = _write_inputs(root, transcript, _manifest())
            client = _fake_client(_full_brief())
            out = S.summarize(cfg, transcript.meeting_id, client=client)
            self.assertEqual(client.calls, [])  # no Claude call on zero turns
            text = out.read_text(encoding="utf-8")
            self.assertIn("no usable speech captured", text)
            self.assertIn("## Open Questions", text)
            self.assertIn(S.ENRICH_PLACEHOLDER, text)


# --------------------------------------------------------------- input validation


class TestInputValidation(unittest.TestCase):
    def test_missing_transcript_raises(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = S.SummarizeConfig(transcripts_dir=td, recordings_dir=td, vault_dir=td)
            with self.assertRaises(S.InputError):
                S.summarize(cfg, "nope", client=_fake_client(_full_brief()))

    def test_failure_leaves_existing_note_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transcript = _transcript()
            cfg = _write_inputs(root, transcript, _manifest())
            out = S.summarize(cfg, transcript.meeting_id, client=_fake_client(_full_brief()))
            before = out.read_text(encoding="utf-8")
            # Corrupt meeting.json, re-run → must fail before touching the note.
            (root / "recordings" / transcript.meeting_id / "meeting.json").write_text(
                "{ not json", encoding="utf-8")
            with self.assertRaises(S.InputError):
                S.summarize(cfg, transcript.meeting_id, client=_fake_client(_full_brief()))
            self.assertEqual(out.read_text(encoding="utf-8"), before)

    def test_bad_brief_shape_raises_claude_error(self):
        with self.assertRaises(S.ClaudeError):
            S._validate_brief_shape({"per_speaker": "not a list", "open_questions": []})
        with self.assertRaises(S.ClaudeError):
            S._validate_brief_shape({"per_speaker": [{"no_speaker": 1}], "open_questions": []})


# ------------------------------------------------------------------- end-to-end fake


class TestEndToEnd(unittest.TestCase):
    def test_fake_client_path_produces_valid_note(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transcript = _transcript()
            cfg = _write_inputs(root, transcript, _manifest())
            client = _fake_client(_full_brief())
            out = S.summarize(cfg, transcript.meeting_id, client=client)

            # Output path matches the contract.
            self.assertEqual(out, root / "vault" / "20-Meetings"
                             / "2026-06-14-01J9ZC8Q9F7Y3K2N5R6T8W0X1Z.md")
            self.assertTrue(out.exists())

            # The client was called once with the roster + transcript text.
            self.assertEqual(len(client.calls), 1)
            self.assertIn("Speaker roster", client.calls[0]["system"])
            self.assertIn("Jane Doe", client.calls[0]["system"])
            self.assertIn("Paul Nathan: Let me walk through", client.calls[0]["transcript_text"])
            self.assertEqual(client.calls[0]["model"], "claude-opus-4-8")

            text = out.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"))
            self.assertIn("## Me (Paul Nathan)", text)
            self.assertIn("## Jane Doe", text)
            self.assertIn("## Speaker_2", text)
            self.assertIn("## Open Questions", text)
            self.assertIn(S.ENRICH_PLACEHOLDER, text)

    def test_schema_file_is_valid_json_and_shaped(self):
        schema = S._load_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("per_speaker", schema["properties"])
        self.assertIn("open_questions", schema["properties"])
        self.assertEqual(schema["required"], ["per_speaker", "open_questions"])


class TestSummarizeBackends(unittest.TestCase):
    """The claude-CLI backend lets `briefly run` summarize with no ANTHROPIC_API_KEY."""

    BRIEF = {"per_speaker": [{"speaker": "Me", "summary": ["hi"], "questions": []}],
             "open_questions": [{"question": "when?"}]}

    @staticmethod
    def _fake_run(stdout, returncode=0, stderr=""):
        from types import SimpleNamespace
        calls = []

        def run(cmd, prompt, timeout):
            calls.append({"cmd": cmd, "prompt": prompt})
            return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

        run.calls = calls
        return run

    def _call(self, client, **kw):
        base = dict(system="sys", transcript_text="t", schema={"x": 1},
                    model="m", max_tokens=1, max_retries=1)
        base.update(kw)
        return client(**base)

    def test_cli_parses_result_envelope(self):
        run = self._fake_run(json.dumps({"result": json.dumps(self.BRIEF)}))
        brief = self._call(S.make_claude_cli_client("claude", run=run))
        self.assertEqual(brief["per_speaker"][0]["speaker"], "Me")
        self.assertIn("-p", run.calls[0]["cmd"])
        self.assertIn("--output-format", run.calls[0]["cmd"])

    def test_cli_strips_prose_and_fences(self):
        wrapped = "Sure!\n```json\n" + json.dumps(self.BRIEF) + "\n```\n"
        run = self._fake_run(json.dumps({"result": wrapped}))
        brief = self._call(S.make_claude_cli_client("claude", run=run), max_retries=0)
        self.assertEqual(brief["open_questions"][0]["question"], "when?")

    def test_cli_retries_then_raises(self):
        run = self._fake_run("not json at all")
        with self.assertRaises(S.ClaudeError):
            self._call(S.make_claude_cli_client("claude", run=run), max_retries=1)
        self.assertEqual(len(run.calls), 2)   # max_retries + 1 attempts

    def test_backend_selection(self):
        import os
        self.assertTrue(callable(S.make_brief_client(S.SummarizeConfig(summarize_backend="cli"))))
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            # auto + no key + bogus claude path -> clear error
            with self.assertRaises(S.ClaudeError):
                S.make_brief_client(S.SummarizeConfig(claude_path="no-such-binary-xyz123"))
            # auto + key present -> a client (the SDK path; no network until called)
            os.environ["ANTHROPIC_API_KEY"] = "sk-test"
            self.assertTrue(callable(S.make_brief_client(S.SummarizeConfig())))
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old


if __name__ == "__main__":
    unittest.main()
