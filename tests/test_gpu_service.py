"""Tests for the optional GPU-container lifecycle (start before / stop after network stages)."""
import unittest
import urllib.error

from briefly.gpu_service import (
    DockerServiceConfig,
    _off_cmd,
    _ready_url_from,
    _up_cmd,
    is_ready,
    managed,
    start,
)


def _runner(rc=0, calls=None):
    def run(cmd, **kw):
        if calls is not None:
            calls.append(cmd)

        class P:
            returncode = rc
            stdout = ""
            stderr = "boom"
        return P()
    return run


class _Resp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestConfig(unittest.TestCase):
    def test_ready_url_derivation(self):
        self.assertEqual(_ready_url_from("http://h:8000/asr"), "http://h:8000/readyz")
        self.assertIsNone(_ready_url_from(""))
        self.assertIsNone(_ready_url_from(None))

    def test_from_env(self):
        env = {"MANAGE_GPU_DOCKER": "1", "GPU_DOCKER_CONTEXT": "paulgaming", "GPU_DOCKER_OFF": "down"}
        cfg = DockerServiceConfig.from_env(asr_url="http://h:8000/asr", env=env)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.context, "paulgaming")
        self.assertEqual(cfg.off_command, "down")
        self.assertEqual(cfg.ready_url, "http://h:8000/readyz")
        self.assertFalse(DockerServiceConfig.from_env(env={}).enabled)

    def test_command_shapes(self):
        cfg = DockerServiceConfig(context="ctx", compose_file="f.yaml", service="whisperx")
        self.assertEqual(_up_cmd(cfg),
                         ["docker", "--context", "ctx", "compose", "-f", "f.yaml", "up", "-d", "whisperx"])
        self.assertEqual(_off_cmd(cfg),
                         ["docker", "--context", "ctx", "compose", "-f", "f.yaml", "stop", "whisperx"])
        down = DockerServiceConfig(compose_file="f.yaml", service="whisperx", off_command="down")
        self.assertEqual(_off_cmd(down), ["docker", "compose", "-f", "f.yaml", "down"])  # no service


class TestIsReady(unittest.TestCase):
    def test_200_is_ready(self):
        self.assertTrue(is_ready(DockerServiceConfig(ready_url="http://x/readyz"),
                                 opener=lambda *a, **k: _Resp(200)))

    def test_503_not_ready(self):
        self.assertFalse(is_ready(DockerServiceConfig(ready_url="http://x/readyz"),
                                  opener=lambda *a, **k: _Resp(503)))

    def test_connection_error_not_ready(self):
        def boom(*a, **k):
            raise urllib.error.URLError("down")
        self.assertFalse(is_ready(DockerServiceConfig(ready_url="http://x/readyz"), opener=boom))

    def test_no_url_not_ready(self):
        self.assertFalse(is_ready(DockerServiceConfig(ready_url=None)))


class TestManaged(unittest.TestCase):
    def test_disabled_is_pure_noop(self):
        calls, readies = [], []
        with managed(DockerServiceConfig(enabled=False), runner=_runner(calls=calls),
                     log=lambda *a: None, ready_fn=lambda c: readies.append(1) or True):
            pass
        self.assertEqual(calls, [])
        self.assertEqual(readies, [])               # disabled → not even a readiness probe

    def test_already_up_left_running(self):
        calls = []
        with managed(DockerServiceConfig(enabled=True, ready_url="http://x/readyz"),
                     runner=_runner(calls=calls), log=lambda *a: None, ready_fn=lambda c: True):
            pass
        self.assertEqual(calls, [])                 # already ready → no up/stop

    def test_down_starts_then_stops(self):
        calls = []
        states = iter([False, True])                # gate False → start; first poll True → ready
        with managed(DockerServiceConfig(enabled=True, ready_url="http://x/readyz",
                                         service="whisperx", context="paulgaming"),
                     runner=_runner(calls=calls), log=lambda *a: None,
                     ready_fn=lambda c: next(states), clock=lambda: 0.0, sleep=lambda s: None):
            pass
        self.assertEqual(len(calls), 2)
        self.assertIn("up", calls[0])
        self.assertIn("whisperx", calls[0])
        self.assertIn("paulgaming", calls[0])
        self.assertIn("stop", calls[1])

    def test_start_failure_raises(self):
        with self.assertRaises(RuntimeError):
            with managed(DockerServiceConfig(enabled=True, ready_url="http://x/readyz"),
                         runner=_runner(rc=1), log=lambda *a: None, ready_fn=lambda c: False,
                         clock=lambda: 0.0, sleep=lambda s: None):
                pass


class TestStart(unittest.TestCase):
    def test_times_out_when_never_ready(self):
        ticks = iter([5, 10, 15, 20])
        with self.assertRaises(RuntimeError):
            start(DockerServiceConfig(ready_url="http://x/readyz", ready_timeout=10, poll_interval=1),
                  runner=_runner(rc=0), log=lambda *a: None, ready_fn=lambda c: False,
                  clock=lambda: next(ticks), sleep=lambda s: None)


if __name__ == "__main__":
    unittest.main()
