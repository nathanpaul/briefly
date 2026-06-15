# Briefly

Capture a meeting on a dedicated soundcard (mic-in = you, line-in = remote via a DAC
line-out), transcribe on a self-hosted Whisper cluster, diarize the remote channel with
pyannote, summarize per-person with Claude, and enrich the notes into an Obsidian vault.

Full design: [PLAN.md](PLAN.md) · architecture: [docs/architecture.md](docs/architecture.md) ·
hardware/test facts + decisions: [knowledge/](knowledge/). **Agents: read `knowledge/` first
(see [CLAUDE.md](CLAUDE.md)).**

## Pipeline

```
capture ─▶ preprocess ─▶ transcribe ─▶ diarize ─▶ merge ─▶ [name speakers] ─▶ summarize ─▶ enrich
 (mic+line)  (AEC/16k)    (Whisper)    (pyannote)  (transcript.json)            (notes.md)   (vault)
```

Each stage reads files and writes files under per-meeting dirs (`recordings/ → processed/ →
transcripts/ → vault/`), keyed by a ULID `meeting_id`. Every stage is independently
re-runnable; `briefly run` skips stages whose output already exists.

## Requirements
- macOS (capture) with the Cubilux CB5 soundcard; `ffmpeg` 8.x at `/opt/homebrew/bin/ffmpeg`.
- Python 3.11+. Install: `pip install -e .` (gives the `briefly` command). Runs on stdlib;
  `summarize` lazily needs `anthropic`, and the optional AEC backend needs
  `webrtc-audio-processing` (preprocess degrades to passthrough without it).
- A reachable **Whisper cluster** and the **pyannote diarization service**
  ([knowledge/cluster/pyannote-deployment.md](knowledge/cluster/pyannote-deployment.md)).
- The `claude` CLI (for `enrich`) and your Obsidian vault (copy/symlink
  [vault-template/](vault-template/); set the `40-Personal` OS guard per the vault README).

## Configure
Set service URLs via flags, env vars (`BRIEFLY_WHISPER_URL`, `BRIEFLY_DIARIZE_URL`,
`BRIEFLY_VAULT_DIR`, …), or a JSON config — see [briefly.example.json](briefly.example.json):

```sh
briefly run --meeting-id <id> --config briefly.json
```

## Run a meeting

```sh
# 1. record (dedicated capture laptop) — meetings of unknown length:
briefly capture start --attendees "Jane Doe,John Smith"   # prints meeting_id; records in background
#    ... the meeting happens ...
briefly capture stop                                       # finalizes recordings/<id>/
#    (or `briefly capture record --duration <sec>` for a fixed length)

# 2. process up to the transcript (stops for naming)
briefly run --meeting-id <id> --config briefly.json
#    → preprocess → transcribe → diarize → merge → transcripts/<id>/transcript.json
#                                                  + a speakers.json stub

# 3. name the speakers: edit transcripts/<id>/speakers.json
#    {"map": {"Me": "Paul Nathan", "Speaker_1": "Jane Doe", ...}}

# 4. summarize + enrich (re-triggerable any time names change)
briefly run --meeting-id <id> --from summarize --to enrich --force --config briefly.json
```

Individual stages are also commands: `briefly {capture,preprocess,transcribe,diarize,merge,
summarize,enrich}` — run `briefly <cmd> --help`.

### Auto-trigger
Instead of running `briefly run` by hand, run the watcher — it processes each meeting the
moment capture finalizes its `meeting.json` (single-worker, resumable, idempotent):
```sh
briefly watch --config briefly.json                  # to=merge (stops for speaker naming)
briefly watch --to enrich --config briefly.json      # fully unattended (keeps Speaker_N labels)
```

## Test
```sh
python3 -m unittest discover -s tests -t .          # stdlib only
pip install -e '.[aec]' && python3 -m unittest discover -s tests -t .   # + numpy AEC tests
```

## Services (k8s-homelab)
- **Diarization:** `https://speaker-diarization.example.io/diarize` (LAN). Dev:
  `kubectl -n speaker-diarization port-forward svc/speaker-diarization 8080:80`. See
  [knowledge/cluster/homelab-services.md](knowledge/cluster/homelab-services.md).
- **Whisper:** `wyoming-whisper` — **Wyoming protocol over TCP :10300, text-only** (not HTTP).

## Status
End-to-end built + tested: capture (`record` + `start`/`stop`), preprocess (**real numpy AEC** —
`[aec]` extra; graceful passthrough without it), **diarization-guided transcription against
wyoming-whisper** (`[whisper]` extra: Wyoming TCP client + VAD; line channel sliced by diarization
turns, mic channel by VAD), diarize client, merge, summarize, enrich, the `run` orchestrator, and
the `watch` auto-trigger. **Before a live run:** install extras (`pip install -e '.[aec,whisper,summarize]'`),
port-forward both services (or use the cluster URLs), and confirm the `claude` CLI + your vault.
**Follow-ups:** capture aggregate-device mode; a finer-grained sync-marker offset; launchd unit for `watch`.
