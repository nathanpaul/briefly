"""Wyoming-protocol Whisper client.

`rhasspy/wyoming-whisper` speaks the Wyoming protocol over TCP :10300 and is text-only.
This sends ONE utterance's 16 kHz mono
PCM16 and returns the transcript text. The `wyoming` package is an optional extra
([whisper]); it is lazy-imported so this module loads without it, and the transcribe call
is injectable in the stage so tests need neither the package nor a server.
"""
from __future__ import annotations

import asyncio

DEFAULT_PORT = 10300
_CHUNK = 2048  # bytes per audio-chunk (1024 samples @ 16-bit mono)


async def _transcribe_async(host: str, port: int, pcm: bytes, rate: int,
                            language: str, timeout: float) -> str:
    from wyoming.asr import Transcribe, Transcript
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.client import AsyncTcpClient

    async def run() -> str:
        async with AsyncTcpClient(host, port) as client:
            await client.write_event(Transcribe(language=language).event())
            await client.write_event(AudioStart(rate=rate, width=2, channels=1).event())
            for i in range(0, len(pcm), _CHUNK):
                await client.write_event(
                    AudioChunk(rate=rate, width=2, channels=1, audio=pcm[i:i + _CHUNK]).event())
            await client.write_event(AudioStop().event())
            while True:
                event = await client.read_event()
                if event is None:
                    return ""
                if Transcript.is_type(event.type):
                    return (Transcript.from_event(event).text or "").strip()

    return await asyncio.wait_for(run(), timeout=timeout)


def transcribe_pcm(pcm: bytes, host: str = "localhost", port: int = DEFAULT_PORT,
                   rate: int = 16000, language: str = "en", timeout: float = 300) -> str:
    """Transcribe one utterance (16 kHz mono PCM16 bytes) via Wyoming. Returns the text."""
    if not pcm:
        return ""
    return asyncio.run(_transcribe_async(host, port, pcm, rate, language, timeout))
