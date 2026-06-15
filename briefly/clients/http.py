"""Minimal stdlib multipart/form-data POST (no `requests`/`httpx` dependency).

Kept tiny and injectable so the transcribe/diarize clients can be unit-tested with a
fake `post` and the package stays dependency-free.
"""
from __future__ import annotations

import urllib.error
import urllib.request
import uuid


class HttpError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def post_multipart(
    url: str,
    files: list[tuple[str, str, bytes, str]],
    fields: list[tuple[str, str]] | None = None,
    timeout: float = 1800,
    headers: dict[str, str] | None = None,
) -> bytes:
    """POST multipart/form-data. `files` = [(name, filename, content, content_type)];
    `fields` = [(name, value)] (a list so repeated keys like `timestamp_granularities[]`
    are allowed). Returns the raw response body. Raises HttpError on a non-2xx status."""
    boundary = "----briefly" + uuid.uuid4().hex
    body = bytearray()

    def w(chunk: str | bytes) -> None:
        body.extend(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)

    for name, value in fields or []:
        w(f"--{boundary}\r\n")
        w(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        w(f"{value}\r\n")
    for name, filename, content, ctype in files:
        w(f"--{boundary}\r\n")
        w(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n')
        w(f"Content-Type: {ctype}\r\n\r\n")
        w(content)
        w("\r\n")
    w(f"--{boundary}--\r\n")

    hdrs = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=bytes(body), method="POST", headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, e.read().decode("utf-8", "replace")) from e
