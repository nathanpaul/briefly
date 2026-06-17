# WhisperX GPU service on a workstation (NVIDIA + Docker Desktop)

Run **transcription** and **diarization** on a machine with an NVIDIA GPU (e.g. an
**RTX 4080 Super**), so a long meeting is processed in a fraction of real time instead of
running CPU Whisper on the cluster. Following the WhisperX example, transcribe→align and
diarize are kept as **two separate steps**, exposed as **two endpoints**:

- `POST /asr` — **transcribe + word-alignment only** (no diarization). Same contract as the
  cluster faster-whisper service, so Briefly's transcribe client is uniform across backends.
- `POST /diarize` — **diarization only** (WhisperX's pyannote `DiarizationPipeline`). Returns
  the **exact same shape as the homelab pyannote `speaker-diarization` service**, so it is a
  **drop-in replacement** for it — the two diarizers are interchangeable.

WhisperX ([m-bain/whisperX](https://github.com/m-bain/whisperX)) chains faster-whisper ASR →
wav2vec2 forced alignment as one stage and pyannote diarization as a separate stage; this
service surfaces those two stages as the two endpoints above. `merge` then assigns the
transcribe segments to the diarization turns. See [the write-up](../../docs/gpu-diarize.md).

## The contract

```
GET  /healthz -> 200 {"status":"ok"}
GET  /readyz  -> 200 {"status":"ready","model":...,"device":"cuda"}   (503 until the ASR
                model is warm; readiness gates on ASR only, so /asr is usable even with no
                HF_TOKEN — /diarize then 503s)

POST /asr  (transcribe + align ONLY — multipart):
   audio        wav file (required)
   language     (default "en")
   model        (optional override; default WHISPERX_MODEL=large-v2)
   (diarize / min_speakers / max_speakers are accepted but IGNORED here — diarization is
    the separate /diarize endpoint below)
 -> 200 JSON:
   { "model", "language", "duration_sec", "processing_sec", "device":"cuda",
     "segments": [ { "start", "end", "text",
                     "words": [ {"start","end","word","score"} ] } ] }      # no speaker labels

POST /diarize  (diarization ONLY — multipart; identical to the homelab pyannote service):
   audio                                  wav file (required)
   num_speakers / min_speakers / max_speakers   (optional int hints)
 -> 200 JSON:
   { "model", "duration_sec", "num_speakers", "processing_sec", "rt_factor",
     "segments": [ {"speaker", "start", "end"} ] }
   503 if no HF_TOKEN (or the gated pyannote model failed to build) — /asr is unaffected.
```

`/asr` returns transcript + word timestamps and **no** speaker labels. `/diarize` returns
who-spoke-when as `SPEAKER_00 … SPEAKER_NN` turns — byte-for-byte the homelab pyannote shape.

## One-time setup (on the GPU machine — Windows shown)

1. **NVIDIA driver** — already installed if you game. Any recent Game Ready / Studio driver
   includes CUDA-on-WSL2; nothing extra to install.
2. **Docker Desktop** with the **WSL2 backend** (the default). Verify the GPU is visible:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```
   You should see the RTX 4080 Super. If not, update Docker Desktop / the driver and ensure
   WSL2 (not Hyper-V) is the backend.
3. **Hugging Face token** *(only needed for `POST /diarize`)* — the diarization model is gated:
   accept the terms for the model you use — <https://hf.co/pyannote/speaker-diarization-community-1>
   (recommended; matches the homelab) or <https://hf.co/pyannote/speaker-diarization-3.1> plus its
   <https://hf.co/pyannote/segmentation-3.0> — make a read token, then:
   ```sh
   cp .env.example .env      # and paste your token into HF_TOKEN
   ```
   Plain transcribe + alignment (`/asr`) work without a token.

## Run

```sh
docker compose up --build -d
docker compose logs -f       # watch for "service ready ... on cuda"
```
First start downloads the models into the `diarize-cache` volume (once): Whisper `large-v2`
(~3 GB), the wav2vec2 align model, and — if a token is set — the diarization model. Check it:
```sh
curl http://localhost:8000/readyz      # {"status":"ready", ..., "device":"cuda"}
```
Quick smoke test (the two endpoints are independent):
```sh
curl -F audio=@meeting.wav http://localhost:8000/asr     | jq .   # transcript + word times
curl -F audio=@meeting.wav http://localhost:8000/diarize | jq .   # who-spoke-when
```

## Open the port + point Briefly at it

- **Windows Firewall** → allow inbound TCP **8000** (so the capture laptop can reach it).
- Note the machine's LAN IP (set a DHCP reservation, or use `<hostname>.local`).
- On the **capture laptop**, in Briefly's `.env`, point the **transcribe** and **diarize**
  steps at this box — they are separate URLs because they are separate steps:
  ```
  TRANSCRIBE_SERVICE_URL=http://<gpu-machine-ip>:8000/asr        # transcribe + align
  DIARIZE_URL=http://<gpu-machine-ip>:8000/diarize     # diarization
  ```
  `/diarize` here is **interchangeable** with the homelab pyannote `speaker-diarization`
  service — point `DIARIZE_URL` at whichever box you prefer; the response is identical.
  That's the entire integration — no port-forward, no gateway.

## Notes
- **LAN only.** Don't forward 8000 to the internet — there's **no auth** on this service.
- **On-demand is fine.** The machine only needs to be on + reachable when you process a
  meeting (a batch step); `restart: unless-stopped` keeps it up while the box is on.
- **Falls back to CPU** (with a warning in the logs, and `compute_type` downgraded to `int8`)
  if CUDA isn't visible, so a misconfigured run still works — just slowly.
- **VRAM:** `large-v2` in float16 plus the diarization pipeline fits comfortably on an
  8 GB+ card; drop to `WHISPERX_MODEL=large-v3`/`medium` or `WHISPERX_COMPUTE_TYPE=int8` if
  you're tight (e.g. running a game at the same time — both share the GPU).
- **Stack** — whisperX 3.8.6 (pyannote.audio 4.0.4, ctranslate2 4.8, faster-whisper 1.2) on
  **torch 2.8 +cu128**. The Dockerfile installs torch from the `cu128` wheel index (torch 2.8 has no
  `+cu124` wheels); bump to `cu126`/`cu129` only if a wheel isn't published for your setup.
- **Diarization model** — `/diarize` loads `pyannote/speaker-diarization-community-1` here (set in
  `.env`, reported as the `model` field); the app default is `speaker-diarization-3.1`. Override with
  `WHISPERX_DIARIZE_MODEL` and accept that model's HF terms.
- **Single worker by design** — WhisperX / pyannote inference isn't thread-safe, so requests
  to **both** endpoints are serialized on one worker thread under a shared lock. Scale out
  with replicas, not threads.
