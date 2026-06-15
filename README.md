# Briefly

> Record a work meeting on a dedicated soundcard and get a clean, **speaker-attributed,
> per-person brief** in your Obsidian vault — transcribed and diarized on your own
> infrastructure, summarized by Claude.

[![tests](https://github.com/nathanpaul/briefly/actions/workflows/tests.yml/badge.svg)](https://github.com/nathanpaul/briefly/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![capture](https://img.shields.io/badge/capture-macOS-lightgrey)
![pipeline](https://img.shields.io/badge/pipeline-validated%20end--to--end-brightgreen)

Briefly splits a meeting into two channels — **mic-in = you**, **line-in = the remote side**
(via a DAC line-out) — so "who said what" falls out of the hardware. Each stage reads a file and
writes a file, so any stage can be re-run in isolation and nothing proprietary leaves your
machines until Claude writes the summary.

**[▶ Run a meeting](docs/running-a-meeting.md)** · [Architecture](docs/architecture.md) ·
[Design notes (PLAN)](PLAN.md) · [Knowledge base](knowledge/) — *agents start at
[CLAUDE.md](CLAUDE.md)*

---

## Pipeline

```mermaid
flowchart LR
    C["capture<br/>mic + line"] --> P["preprocess<br/>AEC · 16 kHz"]
    P --> D["diarize<br/>pyannote"]
    D --> T["transcribe<br/>wyoming-whisper"]
    T --> M["merge<br/>transcript.json"]
    M --> NAME{{name speakers}}
    NAME --> S["summarize<br/>Claude · notes.md"]
    S --> E["enrich<br/>Obsidian vault"]
```

**Diarize runs before transcribe** — wyoming-whisper is text-only, so the line channel is sliced
by the diarization turns and each slice is transcribed (the mic channel is VAD-segmented). Stage
outputs live in per-meeting dirs (`recordings/ → processed/ → transcripts/ → vault/`), keyed by a
ULID `meeting_id`; `briefly run` skips any stage whose output already exists.

## Quickstart

```sh
# on the capture laptop (macOS + ffmpeg)
pip install -e '.[aec,whisper,summarize]'
cp .env.example .env                       # point at your whisper + diarize services

briefly capture start --attendees "Jane Doe,John Smith"   # prints a meeting_id; records detached
#   … the meeting happens …
briefly capture stop                                       # finalizes recordings/<id>/

briefly run                  # preprocess → diarize → transcribe → merge (defaults to last capture)
#   name the speakers in transcripts/<id>/speakers.json:
#   {"map": {"Me": "You", "Speaker_1": "Jane Doe", "Speaker_2": "John Smith"}}
briefly run --from summarize --to enrich --force           # → per-person brief in the vault
```

`briefly run` auto-loads `.env` and defaults to the last captured meeting, so no `--meeting-id` is
needed. Every stage is also its own command (`briefly {capture,preprocess,diarize,transcribe,merge,summarize,enrich}` — add `--help`). Full walkthrough with audio-chain + gain guidance:
**[docs/running-a-meeting.md](docs/running-a-meeting.md)**.

<details>
<summary><b>Or run it fully automatically (watch mode)</b></summary>

```sh
briefly watch                  # processes each new capture up to merge, then stops for naming
briefly watch --to enrich      # fully unattended (keeps Speaker_N labels until you rename + re-run)
```
The watcher is single-worker, resumable, and idempotent — it fires the moment capture finalizes a
meeting's `meeting.json`.
</details>

## Requirements

| | |
|---|---|
| **Capture** | macOS with the Cubilux CB5 soundcard; `ffmpeg` 8.x at `/opt/homebrew/bin/ffmpeg`. |
| **Runtime** | Python 3.11+. Core is **stdlib-only**; `pip install -e '.[aec,whisper,summarize]'` adds `numpy` (real AEC), `wyoming` (STT client), and `anthropic` (Claude). Without the extras, AEC passes through and transcribe/summarize are unavailable. (Or `pip install -r requirements.txt`.) |
| **Services** | A **wyoming-whisper** endpoint (Wyoming/TCP) and a **pyannote diarization** HTTP service — your [homelab](knowledge/cluster/homelab-services.md), or both in [local Docker](docs/local-docker-fallback.md) *(planned)*. |
| **Claude** | The `claude` CLI for `enrich` (uses your Claude Code auth) + `ANTHROPIC_API_KEY` for `summarize`. |
| **Vault** | Copy [vault-template/](vault-template/) and set the `40-Personal` OS guard (see its README). |

## Configuration

`briefly run` / `briefly watch` auto-load a **`.env`** in the working directory (gitignored; copy
[`.env.example`](.env.example)). Real env vars and CLI flags override it.

| Key | Purpose |
|---|---|
| `BRIEFLY_DIARIZE_URL` | pyannote `POST /diarize` (HTTP) |
| `BRIEFLY_WHISPER_HOST` / `BRIEFLY_WHISPER_PORT` | wyoming-whisper (Wyoming/TCP — no HTTP route) |
| `BRIEFLY_VAULT_DIR` / `BRIEFLY_DATA_ROOT` | Obsidian vault + where `recordings/…` live |

A JSON config ([briefly.example.json](briefly.example.json), `--config`) works too.

## Services

- **Diarization** — a pyannote FastAPI service, `POST /diarize` (multipart field `audio`) → `BRIEFLY_DIARIZE_URL`.
- **Whisper** — `wyoming-whisper`, **Wyoming protocol over TCP `:10300`, text-only** (not HTTP) → `BRIEFLY_WHISPER_HOST` / `…_PORT`.
- **No homelab?** Run both in Docker on the laptop — [docs/local-docker-fallback.md](docs/local-docker-fallback.md) *(planned)*.

Homelab specifics: [knowledge/cluster/homelab-services.md](knowledge/cluster/homelab-services.md).

## Testing

```sh
pip install -e '.[aec]'                          # numpy → the real AEC tests run (else skipped)
python3 -m unittest discover -s tests -t .
```

The suite is **fully offline** — whisper, diarization, and Claude are all faked — so it needs no
services and no Docker. [CI](.github/workflows/tests.yml) runs it on macOS (Python 3.11–3.13) on
every push and PR; the ffmpeg-backed capture/preprocess tests use synthetic `lavfi` sources.

## Status & roadmap

**Validated end-to-end on real hardware (2026-06-15):** CB5 capture → preprocess (AEC) → live
pyannote diarization → 3-replica wyoming-whisper transcription → merge → a correct
speaker-attributed transcript.

<details>
<summary><b>Before your first real meeting</b></summary>

- **Lower the mic preamp** so peaks sit −6…−12 dB (clipping is detected and warned).
- **Monitor on closed-back / IEMs** so the remote audio doesn't leak into the "Me" channel. AEC +
  merge de-dup handle direct leakage; room/speaker leakage is worse — see
  [knowledge/test-results/live-capture-2026-06-15.md](knowledge/test-results/live-capture-2026-06-15.md).
</details>

**Follow-ups:** local Docker fallback for the two services (planned) · audio-energy echo de-dup ·
capture aggregate-device mode + sync-marker offset · a launchd unit for `watch`.
