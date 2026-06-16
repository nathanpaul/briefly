# GPU diarization on a workstation (NVIDIA + Docker Desktop)

Run the speaker-diarization service on a machine with an NVIDIA GPU (e.g. an **RTX 4080
Super**) so a long meeting diarizes in **seconds** instead of ~30 minutes. It exposes the same
`POST /diarize` API as the homelab service, so Briefly only needs `BRIEFLY_DIARIZE_URL` pointed
at this box. Going direct (no k8s gateway) also sidesteps the gateway's 504 on long files.

This folder is a self-contained copy of the homelab `speaker-diarization` app with **one**
change: env-driven device selection (`DIARIZE_DEVICE=cuda`). See [the write-up](../../docs/gpu-diarize.md).

## One-time setup (on the GPU machine — Windows shown)

1. **NVIDIA driver** — already installed if you game. Any recent Game Ready / Studio driver
   includes CUDA-on-WSL2; nothing extra to install.
2. **Docker Desktop** with the **WSL2 backend** (the default). Verify the GPU is visible:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```
   You should see the RTX 4080 Super. If not, update Docker Desktop / the driver and ensure
   WSL2 (not Hyper-V) is the backend.
3. **Hugging Face token** — the model is gated: accept the terms at
   <https://hf.co/pyannote/speaker-diarization-community-1>, make a read token, then:
   ```sh
   cp .env.example .env      # and paste your token into HF_TOKEN
   ```

## Run

```sh
docker compose up --build -d
docker compose logs -f       # watch for "pipeline ready ... on cuda"
```
First start downloads the ~1 GB model into the `diarize-cache` volume (once). Check it:
```sh
curl http://localhost:8080/readyz      # {"status":"ready", ..., "device":"cuda"}
```

## Open the port + point Briefly at it

- **Windows Firewall** → allow inbound TCP **8080** (so the capture laptop can reach it).
- Note the machine's LAN IP (set a DHCP reservation, or use `<hostname>.local`).
- On the **capture laptop**, in Briefly's `.env`:
  ```
  BRIEFLY_DIARIZE_URL=http://<gpu-machine-ip>:8080/diarize
  ```
  That's the entire integration — no Briefly code change, no port-forward, no gateway.

## Notes
- **LAN only.** Don't forward 8080 to the internet — there's no auth on this service.
- **On-demand is fine.** The machine only needs to be on + reachable when you process a meeting
  (a batch step); `restart: unless-stopped` keeps it up while the box is on.
- **Falls back to CPU** (with a warning in the logs) if CUDA isn't visible, so a misconfigured
  run still works — just slowly.
- **VRAM:** the pipeline uses well under 2 GB, so it won't disturb gaming (though running both at
  once shares the GPU).
- **Keep in sync:** if the homelab `speaker-diarization/app/main.py` changes, re-copy it here and
  re-apply the device tweak (or upstream the tweak — it's backward-compatible, default CPU).
