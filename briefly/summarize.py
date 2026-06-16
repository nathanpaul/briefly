"""Summarize stage — turn transcript.json (+ speakers.json, meeting.json) into notes.md.

This is the **Claude cloud API** step (Anthropic SDK), NOT Claude Code, and NOT the
vault-aware enrichment (that is the downstream `enrich-meeting` skill). It does
summarize-only: per-person summaries + question extraction, deterministically rendered
into the Obsidian meeting note. It must never write inside the
`<!-- briefly:enrichment:* -->` managed block — that is the enrich stage's territory.

Two-layer design (see docs/summarize-contract.md):
  (a) build a prompt from the named transcript and ask Claude for a STRUCTURED JSON brief
      (schema: briefly/schemas/brief.schema.json); Claude owns the prose.
  (b) a DETERMINISTIC renderer maps that JSON into the notes.md template; the app owns
      all structure (frontmatter, headings, ordering, the managed block).

PRIVACY: transcript **text** leaves the device to the Claude cloud API. Raw audio never
does (it stays on the capture laptop / Whisper cluster). This is a settled decision
(knowledge/decisions/design-decisions.md → reasoning = Claude cloud). The real Claude
client lazy-imports `anthropic` so the renderer + module tests run without the SDK.

CLI:
  briefly summarize --meeting-id <id> [--transcripts-dir transcripts] \
      [--recordings-dir recordings] [--vault-dir vault] [--model claude-opus-4-8] \
      [--config summarize.yaml]

Exit 0 on success; non-zero on validation / API / schema-parse failure (message on
stderr). A non-zero exit must NEVER have modified an existing notes.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from .models import MeetingManifest, SpeakersMap, Transcript

# The structured-output schema Claude is asked to return.
SCHEMA_PATH = Path(__file__).with_name("schemas") / "brief.schema.json"

# Managed enrichment block markers — summarize emits/preserves this verbatim, never
# writes inside it. Must match vault-template/_Templates/meeting.md exactly.
ENRICH_START = "<!-- briefly:enrichment:start -->"
ENRICH_END = "<!-- briefly:enrichment:end -->"
ENRICH_PLACEHOLDER = (
    f"{ENRICH_START}\n"
    "<!-- enrichment will be generated here by the enrich-meeting skill; "
    "this block is replaced on each run -->\n"
    f"{ENRICH_END}"
)


# --------------------------------------------------------------------------- config


@dataclass
class SummarizeConfig:
    """Summarize stage configuration. CLI flags override; a future summarize.yaml may too.

    Default model is `claude-opus-4-8` — judgment-heavy per-person synthesis that runs
    ~once per meeting, and Opus 4.8's 1M context holds long meetings without chunking
    (see docs/summarize-contract.md). Configurable to `claude-sonnet-4-6` (cheaper/faster,
    also 1M context) or `claude-haiku-4-5` (cheapest; 200K context). These current models
    reject `temperature`/`top_p`/`top_k` (400), so repeatability rests on the pinned model
    + a fixed deterministic prompt + the structured schema — NOT on sampling params.
    """

    model: str = "claude-opus-4-8"
    vault_dir: str = "vault"
    meetings_subdir: str = "20-Meetings"
    transcripts_dir: str = "transcripts"
    recordings_dir: str = "recordings"
    max_retries: int = 2
    max_tokens: int = 16000
    # Above this many input tokens, map-reduce the transcript. With a 1M-context model
    # this is rarely hit; chunking is approximate (token≈chars/4, no SDK dependency).
    chunk_threshold_tokens: int = 150000
    # project/proposal are not known at capture; left empty for the human/enrich stage
    # unless provided here. Stored as wikilink strings (e.g. "[[Apollo MOC]]") or "".
    project: str = ""
    proposal: str = ""
    # Claude backend. "auto": use the Anthropic SDK when ANTHROPIC_API_KEY is set, else the
    # local `claude` CLI (Claude Code auth) — so `briefly run` needs no API key when claude is
    # installed. Force with "api" / "cli".
    claude_path: str = "claude"
    summarize_backend: str = "auto"


# ------------------------------------------------------------------------- exceptions


class SummarizeError(Exception):
    """Base error. exit_code drives the process exit; existing notes.md is left intact."""

    exit_code = 1


class InputError(SummarizeError):
    """Missing or invalid required input (transcript.json / meeting.json)."""

    exit_code = 2


class ClaudeError(SummarizeError):
    """Claude refused, errored, or returned output that fails schema validation."""

    exit_code = 3


# ---------------------------------------------------------------------- Claude client


class BriefClient(Protocol):
    """Callable interface for the Claude call. Tests inject a fake; the real one wraps
    the Anthropic SDK. Takes the assembled system prompt + transcript text, returns the
    validated structured brief dict (already schema-conformant)."""

    def __call__(self, *, system: str, transcript_text: str, schema: dict,
                 model: str, max_tokens: int, max_retries: int) -> dict: ...


def _validate_brief_shape(brief: object) -> dict:
    """Minimal structural validation of a brief dict (no jsonschema dependency).

    Guards the renderer against malformed model output. Raises ClaudeError on a shape
    that the renderer can't consume."""
    if not isinstance(brief, dict):
        raise ClaudeError(f"brief is not an object: {type(brief).__name__}")
    if not isinstance(brief.get("per_speaker", []), list):
        raise ClaudeError("brief.per_speaker is not an array")
    if not isinstance(brief.get("open_questions", []), list):
        raise ClaudeError("brief.open_questions is not an array")
    for i, ps in enumerate(brief.get("per_speaker", [])):
        if not isinstance(ps, dict) or "speaker" not in ps:
            raise ClaudeError(f"brief.per_speaker[{i}] missing 'speaker'")
    for i, oq in enumerate(brief.get("open_questions", [])):
        if not isinstance(oq, dict) or "question" not in oq:
            raise ClaudeError(f"brief.open_questions[{i}] missing 'question'")
    return brief


def make_anthropic_client() -> BriefClient:
    """Build the real Claude client. LAZY-imports `anthropic` so importing this module
    (and running the renderer/module tests) does NOT require the SDK installed.

    Uses streaming + get_final_message() per the SDK guidance (long transcripts, high
    max_tokens), adaptive thinking, and `output_config.format` for the structured brief.
    Does NOT pass temperature/top_p/top_k — current models reject them.
    """

    def _call(*, system: str, transcript_text: str, schema: dict,
              model: str, max_tokens: int, max_retries: int) -> dict:
        import anthropic  # lazy: only needed for the real network path

        client = anthropic.Anthropic(max_retries=max_retries)
        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                system=[{
                    "type": "text",
                    "text": system,
                    # Stable instructions/schema first → prompt-cache the prefix.
                    "cache_control": {"type": "ephemeral"},
                }],
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": transcript_text}],
            ) as stream:
                message = stream.get_final_message()
        except anthropic.APIError as e:  # network / 4xx / 5xx after retries
            raise ClaudeError(f"Claude API error: {e}") from e

        if getattr(message, "stop_reason", None) == "refusal":
            raise ClaudeError("Claude refused to summarize this transcript")

        text = next((b.text for b in message.content if getattr(b, "type", None) == "text"), None)
        if not text:
            raise ClaudeError("Claude returned no text content for the structured brief")
        try:
            brief = json.loads(text)
        except json.JSONDecodeError as e:
            raise ClaudeError(f"Claude output is not valid JSON: {e}") from e
        return _validate_brief_shape(brief)

    return _call


def _extract_json(text: str) -> dict:
    """Parse a JSON object from model text that may be wrapped in prose or ``` fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        return json.loads(text[i:j + 1])   # outermost {...}
    raise ClaudeError("no JSON object found in claude CLI output")


def make_claude_cli_client(claude_path: str = "claude", run=None) -> BriefClient:
    """Claude client backed by the local `claude` CLI (Claude Code auth) instead of the Anthropic
    SDK — so `briefly run` needs no ANTHROPIC_API_KEY when claude is installed (same auth `enrich`
    uses). The CLI has no structured-output guarantee, so we ask for JSON-only, extract it, and
    validate; `max_retries` covers a stray bad parse. `run` is injectable for tests."""
    import subprocess

    def _default_run(cmd, prompt, timeout):
        return subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)

    runner = run or _default_run

    def _call(*, system: str, transcript_text: str, schema: dict,
              model: str, max_tokens: int, max_retries: int) -> dict:
        prompt = (
            f"{system}\n\nRespond with ONLY a single JSON object matching this schema — no prose, "
            f"no markdown, no code fences:\n{json.dumps(schema)}\n\nTranscript:\n{transcript_text}"
        )
        cmd = [claude_path, "-p", "--output-format", "json", "--allowedTools", ""]
        if model:
            cmd += ["--model", model]
        last = "claude CLI produced no valid brief"
        for _ in range(max(1, max_retries + 1)):
            try:
                proc = runner(cmd, prompt, 900)
            except FileNotFoundError as e:
                raise ClaudeError(f"claude CLI not found ({claude_path!r}): {e}") from e
            if proc.returncode != 0:
                last = f"claude -p failed (exit {proc.returncode}): {(proc.stderr or '')[:300]}"
                continue
            # `--output-format json` wraps the reply: {... "result": "<text>"}. Fall back to raw.
            text = proc.stdout
            try:
                env = json.loads(proc.stdout)
                if isinstance(env, dict) and "result" in env:
                    text = env["result"]
            except json.JSONDecodeError:
                pass
            try:
                return _validate_brief_shape(_extract_json(text))
            except (ClaudeError, json.JSONDecodeError) as e:
                last = f"claude CLI output was not a valid brief: {e}"
        raise ClaudeError(last)

    return _call


def make_brief_client(cfg: SummarizeConfig) -> BriefClient:
    """Pick the Claude backend: the Anthropic SDK if ANTHROPIC_API_KEY is set, else the local
    `claude` CLI if installed — so summarize needs no API key when claude is available. Force via
    cfg.summarize_backend ('api' | 'cli' | 'auto')."""
    import shutil

    backend = getattr(cfg, "summarize_backend", "auto")
    if backend == "api":
        return make_anthropic_client()
    if backend == "cli":
        return make_claude_cli_client(cfg.claude_path)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return make_anthropic_client()
    if shutil.which(cfg.claude_path):
        return make_claude_cli_client(cfg.claude_path)
    raise ClaudeError(
        "no ANTHROPIC_API_KEY set and the `claude` CLI was not found — set the key, install "
        "Claude Code, or pass --summarize-backend")


# --------------------------------------------------------------------------- loading


def _read_json(path: Path, what: str) -> dict:
    if not path.exists():
        raise InputError(f"required {what} not found: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise InputError(f"could not read {what} at {path}: {e}") from e


def load_inputs(cfg: SummarizeConfig, meeting_id: str) -> tuple[Transcript, MeetingManifest, SpeakersMap | None]:
    """Load + validate the required inputs (transcript.json, meeting.json) and the
    optional speakers.json. Raises InputError on a missing/invalid required input."""
    tdir = Path(cfg.transcripts_dir) / meeting_id
    rdir = Path(cfg.recordings_dir) / meeting_id

    try:
        transcript = Transcript.from_dict(_read_json(tdir / "transcript.json", "transcript.json"))
    except (KeyError, TypeError) as e:
        raise InputError(f"transcript.json is malformed: {e}") from e
    try:
        manifest = MeetingManifest.from_dict(_read_json(rdir / "meeting.json", "meeting.json"))
    except (KeyError, TypeError) as e:
        raise InputError(f"meeting.json is malformed: {e}") from e

    speakers_map: SpeakersMap | None = None
    sm_path = tdir / "speakers.json"
    if sm_path.exists():
        try:
            speakers_map = SpeakersMap.from_dict(_read_json(sm_path, "speakers.json"))
        except (KeyError, TypeError) as e:
            raise InputError(f"speakers.json is malformed: {e}") from e
    return transcript, manifest, speakers_map


def speaker_display_names(transcript: Transcript,
                          speakers_map: SpeakersMap | None = None) -> list[str]:
    """Deterministic ordered list of display names, one per transcript.speakers[] entry
    (Me first, then Speaker_1..N by definition order). Resolves name → label, with
    speakers.json as a fallback when a transcript lacks resolved names.

    The display name is the value that appears in turns[].speaker, so Claude keys its
    per_speaker entries by it. The "Me" speaker resolves to its real name (e.g. "Paul
    Nathan") here — the `## Me (<name>)` heading is applied later in section_headings()."""
    out: list[str] = []
    fallback = speakers_map.map if speakers_map else {}
    for s in transcript.speakers:
        name = s.name or fallback.get(s.label) or s.label
        out.append(name)
    return out


def section_headings(transcript: Transcript,
                     speakers_map: SpeakersMap | None = None) -> dict[str, str]:
    """Map each speaker display name → its rendered `## <heading>` text.

    Per the contract, the Me speaker (speaker_id == "me") renders as `## Me (<name>)`
    when named, else `## Me`. Every other speaker renders under their display name as-is
    (including unmapped `Speaker_N`)."""
    names = speaker_display_names(transcript, speakers_map)
    headings: dict[str, str] = {}
    for s, display in zip(transcript.speakers, names):
        if s.id == "me":
            named = s.name or (speakers_map.map.get(s.label) if speakers_map else None)
            headings[display] = f"Me ({named})" if named and named != "Me" else "Me"
        else:
            headings[display] = display
    return headings


# ----------------------------------------------------------------------- prompt build


_SYSTEM_INSTRUCTIONS = """\
You are the summarize stage of a meeting-notes pipeline. You receive a transcript of a \
business meeting and produce a STRUCTURED brief as JSON (the application renders the \
markdown — you only supply prose).

Produce, conforming to the provided JSON schema:
- per_speaker: exactly one entry per speaker in the roster below, using each speaker's \
  display name VERBATIM as the `speaker` field. For each: `summary` is concise bullet \
  points of what that person said/contributed; `questions` is the questions THAT person \
  raised (empty array if none).
- open_questions: the aggregated questions still open across the meeting, each with an \
  optional `owner` ("us", "them", or a name).
- headline (optional): a single-line summary of the meeting.
- decisions (optional): cross-cutting decisions that were made.

Rules:
- Include every roster speaker, even one still labelled "Speaker_N" (not yet named).
- "Me" is the user/note-taker.
- Do NOT invent content not supported by the transcript. Do NOT add wikilinks or vault \
  references — that is a later stage.
"""

_PARTIAL_NOTE = (
    "NOTE: this transcript is PARTIAL/TRUNCATED (the recording was cut short). Do not "
    "over-claim coverage; summarize only what is present.\n"
)


def build_brief_prompt(transcript: Transcript,
                       speakers_map: SpeakersMap | None = None) -> tuple[str, str]:
    """Return (system, transcript_text).

    The system prompt holds the STABLE instructions + the speaker roster (so Claude knows
    exactly which sections to fill, including Speaker_N). The transcript text is the
    VOLATILE part, sent last as the user message — stable-first for prompt-cache.
    """
    names = speaker_display_names(transcript, speakers_map)
    roster = "\n".join(f"- {n}" for n in names) or "- (no speakers)"
    parts = [_SYSTEM_INSTRUCTIONS, ""]
    if transcript.partial:
        parts.append(_PARTIAL_NOTE)
    parts.append("Speaker roster (use these exact display names):")
    parts.append(roster)
    system = "\n".join(parts)
    transcript_text = transcript.to_text() or "(transcript is empty — no speech captured)"
    return system, transcript_text


def _approx_tokens(text: str) -> int:
    """Cheap token estimate (≈4 chars/token) — avoids an SDK/network dependency for the
    chunking decision. Conservative; the 1M-context default rarely trips the threshold."""
    return len(text) // 4 + 1


# -------------------------------------------------------------------------- reconcile


def reconcile_brief(brief: dict, transcript: Transcript,
                    speakers_map: SpeakersMap | None = None) -> dict:
    """Ensure exactly one per_speaker entry per transcript speaker, in transcript order.

    Back-fills an empty section for any transcript speaker Claude omitted; drops any
    per_speaker entry whose `speaker` is not a known display name. Open questions /
    headline / decisions pass through unchanged."""
    names = speaker_display_names(transcript, speakers_map)
    by_speaker = {ps["speaker"]: ps for ps in brief.get("per_speaker", [])
                  if isinstance(ps, dict) and "speaker" in ps}
    ordered: list[dict] = []
    for name in names:
        ps = by_speaker.get(name)
        if ps is None:
            ordered.append({"speaker": name, "summary": [], "questions": []})
        else:
            ordered.append({
                "speaker": name,
                "summary": list(ps.get("summary", [])),
                "questions": list(ps.get("questions", [])),
            })
    out = dict(brief)
    out["per_speaker"] = ordered
    return out


# ----------------------------------------------------------------------- yaml-ish dump


def _yaml_scalar(value: str) -> str:
    """Emit a frontmatter scalar. Wikilink strings ("[[X]]") and values with special
    chars are double-quoted; plain text is left bare. Empty → empty."""
    if value == "":
        return ""
    if value.startswith("[[") or any(c in value for c in ':"#') or value != value.strip():
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return value


def _yaml_list(items: list[str]) -> str:
    """Emit a frontmatter inline list of (possibly-wikilink) strings."""
    if not items:
        return "[]"
    return "[" + ", ".join('"' + i.replace('"', '\\"') + '"' for i in items) + "]"


# --------------------------------------------------------------------------- renderer


def _title(brief: dict, transcript: Transcript, cfg: SummarizeConfig) -> str:
    headline = (brief.get("headline") or "").strip()
    if headline:
        return headline
    project_plain = cfg.project.strip().strip("[]") if cfg.project else ""
    if project_plain:
        return f"{project_plain} — {transcript.date} meeting"
    return transcript.meeting_id


def _render_speaker_section(ps: dict, headings: dict[str, str] | None = None) -> str:
    name = ps["speaker"]
    # "Me" renders as "Me (<name>)" when Me has a resolved name; bare "Me" otherwise.
    # Other speakers render under their display name (including unmapped "Speaker_N").
    heading = (headings or {}).get(name, name)
    lines = [f"## {heading}", ""]
    summary = [s for s in ps.get("summary", []) if s and s.strip()]
    for bullet in summary:
        lines.append(f"- {bullet.strip()}")
    questions = [q for q in ps.get("questions", []) if q and q.strip()]
    if questions:
        lines.append("- Questions raised:")
        for q in questions:
            lines.append(f"  - {q.strip()}")
    else:
        lines.append("- Questions raised: none.")
    return "\n".join(lines)


def render_notes(brief: dict, transcript: Transcript, manifest: MeetingManifest,
                 cfg: SummarizeConfig, enrich_block: str = ENRICH_PLACEHOLDER,
                 speakers_map: SpeakersMap | None = None) -> str:
    """Pure function: structured brief + inputs → full notes.md body (string).

    The app, not Claude, emits the markdown. Appends `enrich_block` verbatim after
    `## Open Questions` (the placeholder on a first write; the preserved block on re-run).
    """
    headline = (brief.get("headline") or "").strip()
    headings = section_headings(transcript, speakers_map)

    fm = [
        "---",
        "type: meeting",
        f"meeting_id: {transcript.meeting_id}",
        f"date: {transcript.date}",
        f"project: {_yaml_scalar(cfg.project)}",
        f"attendees: {_yaml_list(list(manifest.attendees))}",
        f"proposal: {_yaml_scalar(cfg.proposal)}",
        "status: draft",
        f"partial: {'true' if transcript.partial else 'false'}",
        "tags: [meeting]",
        f"summary: {_yaml_scalar(headline)}",
        "---",
    ]

    body = ["\n".join(fm), "", f"# {_title(brief, transcript, cfg)}", ""]

    for ps in brief.get("per_speaker", []):
        body.append(_render_speaker_section(ps, headings))
        body.append("")

    body.append("## Open Questions")
    open_qs = [oq for oq in brief.get("open_questions", [])
               if isinstance(oq, dict) and (oq.get("question") or "").strip()]
    if open_qs:
        for oq in open_qs:
            q = oq["question"].strip().rstrip(".")
            owner = (oq.get("owner") or "").strip()
            if owner:
                body.append(f"- {q} (owner: {owner}).")
            else:
                body.append(f"- {q}.")
    body.append("")

    body.append(enrich_block)
    body.append("")  # trailing newline at EOF

    return "\n".join(body)


# ------------------------------------------------------------- enrich-block-preserving


def extract_enrich_block(existing: str) -> str | None:
    """Return the managed enrichment block (markers inclusive) from existing note text,
    or None if absent/malformed. This block is the enrich stage's territory and must be
    preserved byte-for-byte across re-runs."""
    start = existing.find(ENRICH_START)
    if start == -1:
        return None
    end = existing.find(ENRICH_END, start)
    if end == -1:
        return None
    return existing[start:end + len(ENRICH_END)]


def write_notes(path: Path, body: str) -> None:
    """Write notes.md atomically (temp file + os.replace), PRESERVING any existing
    enrichment block byte-for-byte.

    `body` is the freshly-rendered note (built with the placeholder block). If a note
    already exists with a non-placeholder enrich block, splice that block back in so the
    enrich stage's content is never clobbered. mkdir -p the parent (20-Meetings/)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = ""
        preserved = extract_enrich_block(existing)
        if preserved is not None and preserved != ENRICH_PLACEHOLDER:
            new_block = extract_enrich_block(body)
            if new_block is not None:
                body = body.replace(new_block, preserved)

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(body)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def output_path(cfg: SummarizeConfig, transcript: Transcript) -> Path:
    """<vault_dir>/<meetings_subdir>/<date>-<meeting_id>.md"""
    return (Path(cfg.vault_dir) / cfg.meetings_subdir
            / f"{transcript.date}-{transcript.meeting_id}.md")


# ----------------------------------------------------------------------------- run


def _load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _empty_brief(reason: str) -> dict:
    """Minimal brief for an empty/near-empty transcript (no Claude call)."""
    return {"headline": reason, "per_speaker": [], "open_questions": []}


def summarize(cfg: SummarizeConfig, meeting_id: str,
              client: BriefClient | None = None) -> Path:
    """Run the stage end to end and return the written note path.

    Idempotent / re-triggerable: regenerates frontmatter + per-person sections +
    Open Questions and PRESERVES the existing enrich block. A non-zero exit (raised
    SummarizeError) never modifies an existing notes.md, because all failures occur
    before write_notes() is reached.
    """
    transcript, manifest, speakers_map = load_inputs(cfg, meeting_id)
    out = output_path(cfg, transcript)

    if not transcript.turns:
        # Empty / near-empty transcript: minimal note, no Claude call.
        brief = _empty_brief("no usable speech captured")
    else:
        if client is None:
            client = make_brief_client(cfg)
        system, transcript_text = build_brief_prompt(transcript, speakers_map)
        schema = _load_schema()

        if _approx_tokens(transcript_text) > cfg.chunk_threshold_tokens:
            brief = _map_reduce(client, transcript, speakers_map, system, schema, cfg)
        else:
            brief = client(
                system=system, transcript_text=transcript_text, schema=schema,
                model=cfg.model, max_tokens=cfg.max_tokens, max_retries=cfg.max_retries,
            )
        brief = _validate_brief_shape(brief)

    brief = reconcile_brief(brief, transcript, speakers_map)
    body = render_notes(brief, transcript, manifest, cfg, speakers_map=speakers_map)
    write_notes(out, body)
    return out


def _chunk_turns(transcript: Transcript, max_chars: int) -> list[list]:
    """Split turns into char-bounded chunks for the map pass."""
    chunks: list[list] = []
    cur: list = []
    size = 0
    for t in transcript.turns:
        line = len(t.text) + 40
        if cur and size + line > max_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += line
    if cur:
        chunks.append(cur)
    return chunks


def _map_reduce(client: BriefClient, transcript: Transcript,
                speakers_map: SpeakersMap | None, system: str, schema: dict,
                cfg: SummarizeConfig) -> dict:
    """Very long transcript: summarize per chunk (map), then merge per-speaker bullets +
    de-dupe open questions (reduce). Per-chunk briefs reuse the same roster system prompt
    so Claude keeps consistent speaker names."""
    max_chars = cfg.chunk_threshold_tokens * 4
    chunks = _chunk_turns(transcript, max_chars)
    partials: list[dict] = []
    for turns in chunks:
        sub = replace(transcript, turns=turns)
        _, text = build_brief_prompt(sub, speakers_map)
        partials.append(client(
            system=system, transcript_text=text, schema=schema,
            model=cfg.model, max_tokens=cfg.max_tokens, max_retries=cfg.max_retries,
        ))

    merged_speakers: dict[str, dict] = {}
    seen_q: set[str] = set()
    open_questions: list[dict] = []
    headline = ""
    for p in partials:
        if not headline and (p.get("headline") or "").strip():
            headline = p["headline"].strip()
        for ps in p.get("per_speaker", []):
            if not isinstance(ps, dict) or "speaker" not in ps:
                continue
            slot = merged_speakers.setdefault(ps["speaker"],
                                              {"speaker": ps["speaker"], "summary": [], "questions": []})
            slot["summary"].extend(ps.get("summary", []))
            slot["questions"].extend(ps.get("questions", []))
        for oq in p.get("open_questions", []):
            if not isinstance(oq, dict) or "question" not in oq:
                continue
            key = oq["question"].strip().lower()
            if key and key not in seen_q:
                seen_q.add(key)
                open_questions.append(oq)
    return {
        "headline": headline,
        "per_speaker": list(merged_speakers.values()),
        "open_questions": open_questions,
    }


# ----------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="briefly summarize",
        description=(
            "Summarize transcript.json (+ speakers.json, meeting.json) into the vault "
            "meeting note notes.md via the Claude cloud API. "
            "PRIVACY: the transcript TEXT is sent to the Claude cloud API; raw audio "
            "never leaves the device. Idempotent — re-running preserves the managed "
            "enrichment block byte-for-byte."
        ),
    )
    p.add_argument("--meeting-id", required=True, help="meeting ULID (per-meeting dir name)")
    p.add_argument("--transcripts-dir", help="dir holding <id>/transcript.json (+ speakers.json)")
    p.add_argument("--recordings-dir", help="dir holding <id>/meeting.json")
    p.add_argument("--vault-dir", help="vault root; note goes to <vault>/20-Meetings/<date>-<id>.md")
    p.add_argument("--model", help="Claude model id (default claude-opus-4-8)")
    p.add_argument("--config", help="path to summarize.yaml (optional)")
    return p


def _load_config_file(path: str) -> dict:
    """Load a tiny flat key: value config file. Tolerates YAML-free environments by doing
    a minimal `key: value` parse (the contract's summarize.yaml is flat scalars)."""
    out: dict = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise InputError(f"could not read config {path}: {e}") from e
    try:
        import yaml  # optional
        loaded = yaml.safe_load(text) or {}
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _config_from(args: argparse.Namespace) -> SummarizeConfig:
    cfg = SummarizeConfig()
    if args.config:
        data = _load_config_file(args.config)
        for k, v in data.items():
            if hasattr(cfg, k) and v not in (None, ""):
                cur = getattr(cfg, k)
                if isinstance(cur, int) and not isinstance(cur, bool):
                    try:
                        v = int(v)
                    except (TypeError, ValueError):
                        continue
                setattr(cfg, k, v)
    # CLI flags override the config file.
    for attr, flag in (("transcripts_dir", "transcripts_dir"),
                       ("recordings_dir", "recordings_dir"),
                       ("vault_dir", "vault_dir"),
                       ("model", "model")):
        val = getattr(args, flag, None)
        if val:
            setattr(cfg, attr, val)
    return cfg


def main(argv: list[str] | None = None) -> int:
    """CLI entry. PRIVACY: transcript text → Claude cloud API; raw audio never leaves the
    device. Returns 0 on success; non-zero (per SummarizeError.exit_code) on failure,
    never having modified an existing notes.md."""
    args = _build_parser().parse_args(argv)
    cfg = _config_from(args)
    try:
        out = summarize(cfg, args.meeting_id)
    except SummarizeError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
