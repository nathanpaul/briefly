"""Merge stage — combine per-channel Whisper + pyannote diarization + optional
human speaker map into the canonical transcript.json (+ transcript.txt).

Implements docs/orchestrator-merge-contract.md. Merging ONLY — no transcription,
diarization, summarization, or auto-naming. Deterministic, file-in/file-out,
stdlib-only (no third-party deps): identical inputs ⇒ byte-identical output.

  briefly merge --meeting-id <id> --transcripts-dir transcripts/<id> \
                --recordings-dir recordings/<id> [--config merge.yaml]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    MeetingManifest,
    Speaker,
    SpeakersMap,
    Transcript,
    Turn,
)

# Per-turn flag vocabulary (contract §Output).
FLAG_OVERLAP = "overlap"
FLAG_ECHO = "possible_echo"
FLAG_UNKNOWN = "unknown_speaker"
FLAG_LOW_CONF = "low_confidence"
FLAG_PARTIAL = "partial"

# Near-equal overlap window for cross-talk detection (a unit overlapping ≥2
# diarization turns within this fraction of its best overlap is flagged).
_OVERLAP_TIE_FRAC = 0.8


class MergeError(Exception):
    """A validation failure. Exit non-zero; leave any existing transcript.json intact."""

    exit_code = 1


class InputError(MergeError):
    """A required input is missing or invalid."""

    exit_code = 2


@dataclass
class MergeConfig:
    """Merge tunables (contract §Config). CLI flags / merge.yaml override."""

    nearest_window_sec: float = 0.5
    echo_overlap_frac: float = 0.6      # legacy; superseded by echo_window_sec (kept for compat)
    echo_text_sim: float = 0.8
    echo_action: str = "flag"          # "flag" | "drop"
    min_turn_sec: float = 0.0
    low_confidence_threshold: float = 0.5
    # echo de-dup (text-only): match a mic turn against LINE text within a time window
    echo_window_sec: float = 2.0        # acoustic-delay / segmentation tolerance
    echo_contain_frac: float = 0.8      # echo if this fraction of the mic turn's words are on the line
    echo_short_max_sec: float = 2.5     # short mic turns (typical leak: "Bye", "Thanks for watching") ...
    echo_short_contain: float = 0.6     # ... only need this much containment

    @classmethod
    def from_dict(cls, d: dict) -> "MergeConfig":
        cfg = cls()
        for k in (
            "nearest_window_sec", "echo_overlap_frac", "echo_text_sim",
            "echo_action", "min_turn_sec", "low_confidence_threshold",
            "echo_window_sec", "echo_contain_frac", "echo_short_max_sec", "echo_short_contain",
        ):
            if k in d and d[k] is not None:
                setattr(cfg, k, d[k])
        return cfg


# --- Merge-internal input shapes (whisper + diarization; schemas in the contract) -----

@dataclass
class Word:
    word: str
    start: float
    end: float
    prob: float | None = None


@dataclass
class WhisperSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "WhisperSegment":
        words = [
            Word(word=w["word"], start=float(w["start"]), end=float(w["end"]),
                 prob=w.get("prob"))
            for w in d.get("words", []) or []
        ]
        return cls(
            id=int(d.get("id", 0)),
            start=float(d["start"]),
            end=float(d["end"]),
            text=d.get("text", ""),
            words=words,
            avg_logprob=d.get("avg_logprob"),
            no_speech_prob=d.get("no_speech_prob"),
        )


@dataclass
class WhisperDoc:
    language: str | None
    duration_sec: float | None
    segments: list[WhisperSegment]

    @classmethod
    def from_dict(cls, d: dict) -> "WhisperDoc":
        if not isinstance(d, dict) or "segments" not in d:
            raise InputError("whisper input missing 'segments'")
        if not isinstance(d["segments"], list):
            raise InputError("whisper 'segments' must be a list")
        try:
            segs = [WhisperSegment.from_dict(s) for s in d["segments"]]
        except (KeyError, TypeError, ValueError) as e:
            raise InputError(f"invalid whisper segment: {e}") from e
        return cls(
            language=d.get("language"),
            duration_sec=d.get("duration_sec"),
            segments=segs,
        )


@dataclass
class DiarTurn:
    speaker: str        # raw "SPEAKER_xx"
    start: float
    end: float


@dataclass
class DiarDoc:
    model: str | None
    duration_sec: float | None
    num_speakers: int | None
    segments: list[DiarTurn]

    @classmethod
    def from_dict(cls, d: dict) -> "DiarDoc":
        if not isinstance(d, dict) or "segments" not in d:
            raise InputError("diarization input missing 'segments'")
        if not isinstance(d["segments"], list):
            raise InputError("diarization 'segments' must be a list")
        try:
            segs = [
                DiarTurn(speaker=str(s["speaker"]), start=float(s["start"]),
                         end=float(s["end"]))
                for s in d["segments"]
            ]
        except (KeyError, TypeError, ValueError) as e:
            raise InputError(f"invalid diarization segment: {e}") from e
        return cls(
            model=d.get("model"),
            duration_sec=d.get("duration_sec"),
            num_speakers=d.get("num_speakers"),
            segments=segs,
        )


# --- Small numeric / text helpers -----------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the temporal intersection of [a0,a1] and [b0,b1] (≥0)."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _avg_logprob_to_conf(avg_logprob: float | None) -> float | None:
    """Map Whisper avg_logprob (≤0) to a 0–1 confidence via exp; None passes through."""
    if avg_logprob is None:
        return None
    import math
    return round(math.exp(avg_logprob), 4)


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def text_similarity(a: str, b: str) -> float:
    """Token-set ratio in [0,1] (Jaccard over word tokens). Deterministic, stdlib-only."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _containment(a: str, b: str) -> float:
    """Fraction of a's word-tokens that also appear in b (0..1). Asymmetric: a short `a`
    fully present in a longer `b` -> ~1.0, which Jaccard would understate."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


# --- Internal working representation for a line unit (segment or word-run) -------------

@dataclass
class _LineUnit:
    start: float
    end: float
    text: str
    raw_speaker: str | None          # "SPEAKER_xx" or None (unknown)
    diar_conf: float | None          # overlap fraction in [0,1]
    confidence: float | None
    flags: list[str] = field(default_factory=list)


# --- Diarization assignment -----------------------------------------------------------

def _assign_unit(start: float, end: float, diar: list[DiarTurn],
                 cfg: MergeConfig) -> tuple[str | None, float | None, list[str], bool]:
    """Assign a [start,end] unit to a diarization turn by max overlap.

    Returns (raw_speaker | None, diar_conf | None, flags, used_nearest).
    Tie-break: earlier-starting diar turn, then lexicographically smaller speaker.
    """
    dur = max(0.0, end - start)
    flags: list[str] = []

    overlaps: list[tuple[float, float, str]] = []  # (overlap, diar.start, speaker)
    for dt in diar:
        ov = _overlap(start, end, dt.start, dt.end)
        if ov > 0.0:
            overlaps.append((ov, dt.start, dt.speaker))

    if overlaps:
        # Best by (overlap desc, start asc, speaker asc).
        best = max(overlaps, key=lambda o: (o[0], -o[1], _neg_str(o[2])))
        best_ov = best[0]
        speaker = best[2]
        # Cross-talk: ≥2 distinct speakers near-equally overlapping.
        near = {o[2] for o in overlaps if o[0] >= _OVERLAP_TIE_FRAC * best_ov}
        if len(near) >= 2:
            flags.append(FLAG_OVERLAP)
        diar_conf = round(best_ov / dur, 4) if dur > 0 else 1.0
        return speaker, diar_conf, flags, False

    # No overlap → nearest within window.
    best_gap = None
    best_choice: tuple[float, str] | None = None  # (diar.start, speaker)
    for dt in diar:
        if end <= dt.start:
            gap = dt.start - end
        elif dt.end <= start:
            gap = start - dt.end
        else:  # shouldn't happen (would have overlapped)
            gap = 0.0
        if gap <= cfg.nearest_window_sec:
            key = (dt.start, dt.speaker)
            if best_gap is None or gap < best_gap or (
                gap == best_gap and key < (best_choice or (float("inf"), ""))
            ):
                best_gap = gap
                best_choice = key
    if best_choice is not None:
        return best_choice[1], 0.0, flags, True

    flags.append(FLAG_UNKNOWN)
    return None, None, flags, False


def _neg_str(s: str) -> str:
    """Sort key helper: invert string ordering so `max` picks the lexicographically
    smaller speaker as the final tie-break (after overlap/start)."""
    return "".join(chr(0x10FFFF - ord(c)) for c in s)


def _split_segment_by_words(seg: WhisperSegment, diar: list[DiarTurn],
                            cfg: MergeConfig, seg_conf: float | None) -> list[_LineUnit]:
    """Per-word assignment, then merge contiguous same-speaker runs into units."""
    assigned: list[tuple[Word, str | None, float | None, list[str]]] = []
    for w in seg.words:
        raw, conf, flags, _ = _assign_unit(w.start, w.end, diar, cfg)
        assigned.append((w, raw, conf, flags))

    units: list[_LineUnit] = []
    cur_words: list[Word] = []
    cur_speaker: str | None = None
    cur_flags: set[str] = set()
    cur_diar_sum = 0.0
    cur_diar_n = 0

    def flush() -> None:
        if not cur_words:
            return
        text = "".join(w.word for w in cur_words).strip()
        if not text:
            return
        start = cur_words[0].start
        end = cur_words[-1].end
        diar_conf = round(cur_diar_sum / cur_diar_n, 4) if cur_diar_n else None
        units.append(_LineUnit(
            start=start, end=end, text=text, raw_speaker=cur_speaker,
            diar_conf=diar_conf, confidence=seg_conf, flags=sorted(cur_flags),
        ))

    for w, raw, conf, flags in assigned:
        if cur_words and raw != cur_speaker:
            flush()
            cur_words = []
            cur_flags = set()
            cur_diar_sum = 0.0
            cur_diar_n = 0
        cur_speaker = raw
        cur_words.append(w)
        cur_flags.update(flags)
        if conf is not None:
            cur_diar_sum += conf
            cur_diar_n += 1
    flush()
    return units


def _line_units(doc: WhisperDoc, diar: list[DiarTurn], cfg: MergeConfig) -> list[_LineUnit]:
    """Turn each line whisper segment into one or more assigned line units."""
    units: list[_LineUnit] = []
    for seg in doc.segments:
        seg_conf = _avg_logprob_to_conf(seg.avg_logprob)
        if seg.words:
            units.extend(_split_segment_by_words(seg, diar, cfg, seg_conf))
        else:
            text = seg.text.strip()
            if not text:
                continue
            raw, diar_conf, flags, _ = _assign_unit(seg.start, seg.end, diar, cfg)
            units.append(_LineUnit(
                start=seg.start, end=seg.end, text=text, raw_speaker=raw,
                diar_conf=diar_conf, confidence=seg_conf, flags=sorted(set(flags)),
            ))
    return units


# --- Stable Speaker_N numbering -------------------------------------------------------

def _number_speakers(units: list[_LineUnit]) -> dict[str, int]:
    """Map raw SPEAKER_xx → N by first appearance on the common timeline.

    Order by (unit.start, raw_speaker) so identical inputs always number the same way.
    `unknown` (raw_speaker is None) is not numbered.
    """
    order: list[tuple[float, str]] = sorted(
        {(u.start, u.raw_speaker) for u in units if u.raw_speaker is not None}
    )
    mapping: dict[str, int] = {}
    n = 0
    for _, raw in order:
        if raw not in mapping:
            n += 1
            mapping[raw] = n
    return mapping


# --- Echo / leakage de-dupe -----------------------------------------------------------

def _line_context(mic: Turn, line_turns: list[Turn], window: float) -> str:
    """Concatenated text of LINE turns within `window` seconds of the mic turn."""
    return " ".join(lt.text for lt in line_turns
                    if lt.end >= mic.start - window and lt.start <= mic.end + window)


def _apply_echo(turns: list[Turn], cfg: MergeConfig) -> tuple[list[Turn], int]:
    """Flag (or drop) mic turns that look like residual LINE leakage on the "Me" channel.

    A mic turn is leakage when its words appear on the LINE channel within a short time
    window (`echo_window_sec`, tolerant of the acoustic delay + differing segment boundaries
    that defeated strict overlap): either high token-set similarity OR high CONTAINMENT of the
    mic turn's words in the nearby line text. Short mic turns ("Bye", "Thanks for watching" —
    the typical leak) use a lenient containment bar; longer turns need a strong match so
    genuine simultaneous speech is preserved. Returns (turns, dropped_count).

    Text-only by design (merge sees transcripts, not audio): leakage with NO matching line
    transcript cannot be caught here — that needs audio-energy de-dup in an earlier stage.
    """
    line_turns = [t for t in turns if t.channel == "line"]
    kept: list[Turn] = []
    dropped = 0
    for t in turns:
        if t.channel != "mic":
            kept.append(t)
            continue
        dur = max(0.0, t.end - t.start)
        is_echo = False
        if dur > 0 and _tokens(t.text):
            ctx = _line_context(t, line_turns, cfg.echo_window_sec)
            if ctx:
                sim = text_similarity(t.text, ctx)
                cont = _containment(t.text, ctx)
                if dur <= cfg.echo_short_max_sec:
                    is_echo = cont >= cfg.echo_short_contain or sim >= cfg.echo_text_sim * 0.7
                else:
                    is_echo = sim >= cfg.echo_text_sim or cont >= cfg.echo_contain_frac
        if is_echo:
            if cfg.echo_action == "drop":
                dropped += 1
                continue
            if FLAG_ECHO not in t.flags:
                t.flags.append(FLAG_ECHO)
        kept.append(t)
    return kept, dropped


# --- The pure merge -------------------------------------------------------------------

def merge(
    manifest: MeetingManifest,
    mic: WhisperDoc,
    line: WhisperDoc,
    diar: DiarDoc,
    speakers_map: SpeakersMap | None = None,
    cfg: MergeConfig | None = None,
    generated_at: str | None = None,
) -> Transcript:
    """Pure merge (no disk I/O). Implements the contract's 10-step algorithm.

    Returns a Transcript; deterministic for given inputs (pass `generated_at` to pin it).
    """
    cfg = cfg or MergeConfig()
    warnings: list[str] = []

    mic_off = manifest.channels["mic"].start_offset_sec if "mic" in manifest.channels else 0.0
    line_off = manifest.channels["line"].start_offset_sec if "line" in manifest.channels else 0.0

    # Step 2: align line timeline (offset applied to whisper + diarization line units).
    diar_turns = [DiarTurn(d.speaker, d.start + line_off, d.end + line_off)
                  for d in diar.segments]
    # Sort diarization deterministically (start, then speaker) for stable tie-breaks.
    diar_turns.sort(key=lambda d: (d.start, d.speaker, d.end))

    line_shifted = WhisperDoc(
        language=line.language, duration_sec=line.duration_sec,
        segments=[
            WhisperSegment(
                id=s.id, start=s.start + line_off, end=s.end + line_off, text=s.text,
                words=[Word(w.word, w.start + line_off, w.end + line_off, w.prob)
                       for w in s.words],
                avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob,
            )
            for s in line.segments
        ],
    )

    # Step 4: assign line units to diarization speakers.
    units = _line_units(line_shifted, diar_turns, cfg)

    n_unknown = sum(1 for u in units if u.raw_speaker is None)
    if n_unknown:
        warnings.append(
            f"{n_unknown} line unit(s) had no diarization turn within "
            f"{cfg.nearest_window_sec}s; labeled unknown"
        )

    # Step 5: stable Speaker_N numbering by first appearance.
    numbering = _number_speakers(units)

    # Build the speakers[] list. "Me" always present (mic channel).
    speakers: list[Speaker] = [
        Speaker(id="me", label="Me", channel="mic", source="channel")
    ]
    for raw, n in sorted(numbering.items(), key=lambda kv: kv[1]):
        speakers.append(
            Speaker(id=f"s{n}", label=f"Speaker_{n}", channel="line",
                    source="diarization")
        )
    has_unknown = any(u.raw_speaker is None for u in units)
    if has_unknown:
        speakers.append(
            Speaker(id="unknown", label="unknown", channel="line", source="diarization")
        )

    # Step 3 + 6: mic turns + line turns → merged, sorted, indexed.
    raw_turns: list[Turn] = []

    for seg in mic.segments:
        text = seg.text.strip()
        if not text:
            continue
        conf = _avg_logprob_to_conf(seg.avg_logprob)
        flags: list[str] = []
        if conf is not None and conf < cfg.low_confidence_threshold:
            flags.append(FLAG_LOW_CONF)
        raw_turns.append(Turn(
            index=-1, speaker_id="me", speaker="Me", channel="mic",
            start=round(seg.start + mic_off, 6), end=round(seg.end + mic_off, 6),
            text=text, confidence=conf, diarization_confidence=None, flags=flags,
        ))

    for u in units:
        if u.raw_speaker is None:
            sid, label = "unknown", "unknown"
        else:
            n = numbering[u.raw_speaker]
            sid, label = f"s{n}", f"Speaker_{n}"
        flags = list(u.flags)
        if u.confidence is not None and u.confidence < cfg.low_confidence_threshold:
            if FLAG_LOW_CONF not in flags:
                flags.append(FLAG_LOW_CONF)
        raw_turns.append(Turn(
            index=-1, speaker_id=sid, speaker=label, channel="line",
            start=round(u.start, 6), end=round(u.end, 6), text=u.text,
            confidence=u.confidence, diarization_confidence=u.diar_conf,
            flags=sorted(set(flags)) if flags else [],
        ))

    # Sort: start asc; tie-break mic before line; then start of stable original order.
    _chan_rank = {"mic": 0, "line": 1}
    raw_turns.sort(key=lambda t: (t.start, _chan_rank.get(t.channel, 2), t.end, t.text))

    # Step 7: echo / leakage de-dupe (operates on sorted turns).
    raw_turns, dropped = _apply_echo(raw_turns, cfg)
    if dropped:
        warnings.append(f"dropped {dropped} mic turn(s) as possible echo (echo_action=drop)")

    # Re-index after any drops.
    for i, t in enumerate(raw_turns):
        t.index = i

    # Step 8: apply speakers.json (names + corrections), recompute display speaker.
    if speakers_map is not None:
        _apply_speakers(speakers, raw_turns, speakers_map, warnings)
    else:
        # No map: display value = label (already set for line; mic stays "Me").
        pass

    # Edge case: partial meeting → set partial + flag last turn.
    partial = bool(manifest.partial)
    if partial and raw_turns:
        last = raw_turns[-1]
        if FLAG_PARTIAL not in last.flags:
            last.flags.append(FLAG_PARTIAL)
            last.flags = sorted(set(last.flags))

    models = {
        "transcription": _model_hint(mic, line),
        "diarization": diar.model,
    }

    return Transcript(
        meeting_id=manifest.meeting_id,
        date=manifest.date,
        generated_at=generated_at if generated_at is not None else _utcnow(),
        partial=partial,
        models=models,
        speakers=speakers,
        turns=raw_turns,
        warnings=warnings,
    )


def _model_hint(mic: WhisperDoc, line: WhisperDoc) -> str | None:
    """Whisper docs carry no model field in the contract; leave null unless given."""
    return None


def _apply_speakers(speakers: list[Speaker], turns: list[Turn],
                    sm: SpeakersMap, warnings: list[str]) -> None:
    """Step 8: set speaker names from map, apply corrections, recompute display value."""
    label_to_speaker = {s.label: s for s in speakers}
    id_to_speaker = {s.id: s for s in speakers}

    # 8a: names from map (keyed by label, e.g. "Me", "Speaker_1").
    for label, name in sm.map.items():
        sp = label_to_speaker.get(label)
        if sp is not None:
            sp.name = name or None

    # 8b: corrections — reassign any turn overlapping a window to `to` (a label).
    for corr in sm.corrections:
        try:
            cs, ce = float(corr["start"]), float(corr["end"])
            to_label = corr["to"]
        except (KeyError, TypeError, ValueError):
            warnings.append(f"skipped malformed correction: {corr!r}")
            continue
        target = label_to_speaker.get(to_label)
        if target is None:
            warnings.append(f"correction target {to_label!r} not a known speaker; skipped")
            continue
        for t in turns:
            if _overlap(t.start, t.end, cs, ce) > 0.0:
                t.speaker_id = target.id
                t.channel = target.channel

    # 8c: recompute every turn's display value (name if mapped, else label).
    for t in turns:
        sp = id_to_speaker.get(t.speaker_id)
        if sp is not None:
            t.speaker = sp.name if sp.name else sp.label


# --- Disk I/O + validation ------------------------------------------------------------

def _load_json(path: Path, what: str) -> dict:
    if not path.exists():
        raise InputError(f"required input missing: {what} ({path})")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise InputError(f"invalid {what} ({path}): {e}") from e


def _build_stub(transcript: Transcript, attendees: list[str]) -> SpeakersMap:
    """A speakers.json stub: every detected speaker with an empty name + roster hint."""
    stub_map = {s.label: "" for s in transcript.speakers if s.label != "unknown"}
    sm = SpeakersMap(meeting_id=transcript.meeting_id, map=stub_map, corrections=[])
    # Attendee roster recorded as a comment-ish field the human can consult.
    d = sm.to_dict()
    d["_attendees_hint"] = list(attendees)
    sm._stub_dict = d  # type: ignore[attr-defined]
    return sm


def _write_stub(sm: SpeakersMap, path: Path) -> None:
    d = getattr(sm, "_stub_dict", None) or sm.to_dict()
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)
        fh.flush()
    import os
    os.replace(tmp, path)


def run(meeting_id: str, transcripts_dir: str | Path, recordings_dir: str | Path,
        cfg: MergeConfig | None = None,
        generated_at: str | None = None) -> tuple[Transcript, list[str]]:
    """Load inputs, merge, and write transcript.json + transcript.txt (+ speakers stub).

    Returns (transcript, warnings). Raises InputError on a missing/invalid required input
    WITHOUT touching any existing transcript.json.
    """
    cfg = cfg or MergeConfig()
    tdir = Path(transcripts_dir)
    rdir = Path(recordings_dir)

    # Step 1: load + validate required inputs (before touching any output).
    manifest_d = _load_json(rdir / "meeting.json", "meeting.json")
    try:
        manifest = MeetingManifest.from_dict(manifest_d)
    except (KeyError, TypeError) as e:
        raise InputError(f"invalid meeting.json: {e}") from e
    if manifest.meeting_id != meeting_id:
        raise InputError(
            f"meeting_id mismatch: arg {meeting_id!r} != meeting.json "
            f"{manifest.meeting_id!r}"
        )

    mic = WhisperDoc.from_dict(_load_json(tdir / "mic.whisper.json", "mic.whisper.json"))
    line = WhisperDoc.from_dict(_load_json(tdir / "line.whisper.json", "line.whisper.json"))
    diar = DiarDoc.from_dict(_load_json(tdir / "line.diarization.json", "line.diarization.json"))

    speakers_path = tdir / "speakers.json"
    speakers_map = None
    if speakers_path.exists():
        try:
            speakers_map = SpeakersMap.read(speakers_path)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            raise InputError(f"invalid speakers.json ({speakers_path}): {e}") from e

    transcript = merge(manifest, mic, line, diar, speakers_map, cfg, generated_at)

    # Steps 9: emit outputs (only after a clean merge).
    transcript.write(tdir / "transcript.json")
    (tdir / "transcript.txt").write_text(transcript.to_text(), encoding="utf-8")

    # Stub written only if speakers.json was absent.
    if speakers_map is None:
        stub = _build_stub(transcript, manifest.attendees)
        _write_stub(stub, speakers_path)
        transcript.warnings.append(
            "speakers.json absent; wrote a stub — fill names and re-run to apply"
        )

    return transcript, transcript.warnings


# --- YAML-ish / JSON config loading (stdlib only) -------------------------------------

def _load_config_file(path: Path) -> dict:
    """Load a merge config. Accepts JSON; falls back to a tiny flat key:value parser
    for simple merge.yaml files (stdlib-only — no PyYAML dependency)."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    out: dict = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip("'\"")
        if not key:
            continue
        if val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        else:
            try:
                out[key] = float(val) if "." in val else int(val)
            except ValueError:
                out[key] = val
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="briefly merge",
        description="merge per-channel transcripts + diarization → transcript.json",
    )
    p.add_argument("--meeting-id", required=True)
    p.add_argument("--transcripts-dir", required=True)
    p.add_argument("--recordings-dir", required=True)
    p.add_argument("--config", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = MergeConfig()
    if args.config:
        cpath = Path(args.config)
        if not cpath.exists():
            print(f"error: config not found: {cpath}", file=sys.stderr)
            return 2
        try:
            cfg = MergeConfig.from_dict(_load_config_file(cpath))
        except OSError as e:
            print(f"error: cannot read config {cpath}: {e}", file=sys.stderr)
            return 2
    try:
        transcript, warnings = run(
            args.meeting_id, args.transcripts_dir, args.recordings_dir, cfg
        )
    except MergeError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code

    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"merged {len(transcript.turns)} turns, {len(transcript.speakers)} speakers "
          f"→ {Path(args.transcripts_dir) / 'transcript.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
