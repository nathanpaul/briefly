# Briefly

Capture a meeting on a dedicated soundcard (mic-in = you, line-in = remote via a DAC
line-out), transcribe on a self-hosted Whisper cluster, diarize the remote channel with
pyannote, summarize per-person with Claude, and enrich the notes into an Obsidian vault.

Run a meeting: **[docs/running-a-meeting.md](docs/running-a-meeting.md)** · full design:
[PLAN.md](PLAN.md) · architecture: [docs/architecture.md](docs/architecture.md) · hardware/test
facts + decisions: [knowledge/](knowledge/). **Agents: read `knowledge/` first (see
[CLAUDE.md](CLAUDE.md)).**

## Pipeline

```
capture ─▶ preprocess ─▶ diarize ─▶ transcribe ─▶ merge ─▶ [name speakers] ─▶ summarize ─▶ enrich
 (mic+line)  (AEC/16k)  (pyannote)  (Wyoming, per turn)  (transcript.json)     (notes.md)    (vault)
```

Diarize runs **before** transcribe: wyoming-whisper is text-only, so the line channel is sliced
by the diarization turns and each slice transcribed (the mic channel is VAD-segmented).

Each stage reads files and writes files under per-meeting dirs (`recordings/ → processed/ →
transcripts/ → vault/`), keyed by a ULID `meeting_id`. Every stage is independently
re-runnable; `briefly run` skips stages whose output already exists.

## Requirements
- macOS (capture) with the Cubilux CB5 soundcard; `ffmpeg` 8.x at `/opt/homebrew/bin/ffmpeg`.
- Python 3.11+. **Install:** `pip install -e '.[aec,whisper,summarize]'` — gives the `briefly`
  command + `numpy` (real AEC), `wyoming` (the STT client), and `anthropic` (Claude). The core is
  stdlib-only; without the extras, AEC passes through and transcribe/summarize are unavailable.
  (Or `pip install -r requirements.txt` for just the libraries.)
- Services: a **wyoming-whisper** endpoint (Wyoming/TCP) and a **pyannote diarization** HTTP
  service — your homelab ([knowledge/cluster/homelab-services.md](knowledge/cluster/homelab-services.md)),
  or run both locally in Docker ([docs/local-docker-fallback.md](docs/local-docker-fallback.md), planned).
- The `claude` CLI (for `enrich`, uses your Claude Code auth) + `ANTHROPIC_API_KEY` (for
  `summarize`). Your Obsidian vault: copy [vault-template/](vault-template/) and set the
  `40-Personal` OS guard (see its README).

## Configure
`briefly run` / `briefly watch` auto-load a **`.env`** in the working directory (gitignored;
copy [`.env.example`](.env.example)). Real env vars and CLI flags override it. Keys:
`BRIEFLY_DIARIZE_URL` (HTTP), `BRIEFLY_WHISPER_HOST` + `BRIEFLY_WHISPER_PORT` (Wyoming/TCP —
wyoming-whisper has no HTTP route), `BRIEFLY_VAULT_DIR`, `BRIEFLY_DATA_ROOT`, … A JSON config
([briefly.example.json](briefly.example.json), `--config`) also works.

## Run a meeting

```sh
# 1. record (dedicated capture laptop) — meetings of unknown length:
briefly capture start --attendees "Jane Doe,John Smith"   # prints meeting_id; records in background
#    ... the meeting happens ...
briefly capture stop                                       # finalizes recordings/<id>/
#    (or `briefly capture record --duration <sec>` for a fixed length)

# 2. process to a transcript (defaults to the last captured meeting; stops for naming)
briefly run
#    → preprocess → diarize → transcribe → merge → transcripts/<id>/transcript.json + speakers.json stub

# 3. name the speakers: edit transcripts/<id>/speakers.json
#    {"map": {"Me": "You", "Speaker_1": "Jane Doe", ...}}

# 4. summarize + enrich (re-triggerable any time names change)
briefly run --from summarize --to enrich --force
```

`briefly run` auto-loads `.env` and defaults to the last captured meeting — no `--meeting-id` /
`--config` needed once `.env` is set. Individual stages are also commands:
`briefly {capture,preprocess,transcribe,diarize,merge,summarize,enrich}` — run `briefly <cmd> --help`.

### Auto-trigger
Instead of running `briefly run` by hand, run the watcher — it processes each meeting the
moment capture finalizes its `meeting.json` (single-worker, resumable, idempotent):
```sh
briefly watch                  # to=merge (stops for speaker naming)
briefly watch --to enrich      # fully unattended (keeps Speaker_N labels)
```

## Test
```sh
python3 -m unittest discover -s tests -t .          # stdlib only
pip install -e '.[aec]' && python3 -m unittest discover -s tests -t .   # + numpy AEC tests
```

## Services
Set both in `.env` (see [`.env.example`](.env.example)); homelab specifics in
[knowledge/cluster/homelab-services.md](knowledge/cluster/homelab-services.md).
- **Diarization:** a pyannote FastAPI service, `POST /diarize` (multipart field `audio`) →
  `BRIEFLY_DIARIZE_URL`.
- **Whisper:** `wyoming-whisper` — **Wyoming protocol over TCP :10300, text-only** (not HTTP) →
  `BRIEFLY_WHISPER_HOST` / `BRIEFLY_WHISPER_PORT`.
- **Local fallback (no homelab):** run both in Docker on the laptop —
  [docs/local-docker-fallback.md](docs/local-docker-fallback.md) (planned).

## Status
**Validated end-to-end on real hardware (2026-06-15):** CB5 capture → preprocess (AEC) → live
pyannote diarization → 3-replica wyoming-whisper transcription → merge → a correct
speaker-attributed transcript. **126 tests passing.** `briefly run` defaults to the last
captured meeting (`recordings/.last-meeting-id`).

Before your first real meeting: **lower the mic preamp** so peaks sit −6…−12 dB (clipping is
detected + warned), and **monitor on closed-back / IEMs** so the remote audio doesn't leak into
the "Me" channel (AEC + merge de-dup handle direct leakage; room/speaker leakage is worse — see
[knowledge/test-results/live-capture-2026-06-15.md](knowledge/test-results/live-capture-2026-06-15.md)).

**Follow-ups:** local Docker fallback for the two services (planned); audio-energy echo de-dup;
capture aggregate-device mode + sync-marker offset; a launchd unit for `watch`.
