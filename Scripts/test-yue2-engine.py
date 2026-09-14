#!/usr/bin/env python3
"""Development-only tests of the retired Python reference worker."""
import importlib.util
import json
from pathlib import Path
import queue
import re
import tempfile
import threading
import time
import unittest
import uuid

WORKER = Path(__file__).resolve().parents[1] / "Scripts/yue2-reference-worker.py"
spec = importlib.util.spec_from_file_location("yue2_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Output:
    def __init__(self):
        self.events = queue.Queue()

    def write(self, value):
        self.events.put(json.loads(value))

    def flush(self):
        pass


class FakeBackend:
    def __init__(self):
        self.requests = []
        self.block = False

    def generate(self, request, cancelled, progress):
        self.requests.append(request)
        progress("Generating music", 0.25)
        while self.block:
            if cancelled():
                raise InterruptedError("Cancelled")
            time.sleep(0.005)
        request["output"].write_bytes(b"fake audio")
        return 0.75

    def close(self):
        pass


class WorkerTests(unittest.TestCase):
    def test_protocol_version_matches_swift_client(self):
        source = (WORKER.parents[1] / "Packages/H3ddleKit/Sources/H3ddleEngineProtocol/EngineProtocol.swift").read_text()
        version = re.search(r"currentVersion\s*=\s*(\d+)", source)
        self.assertIsNotNone(version)
        self.assertEqual(worker.PROTOCOL_VERSION, int(version.group(1)))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="yue2 worker ")
        self.root = Path(self.directory.name)
        self.backend = FakeBackend()
        self.output = Output()
        self.server = worker.Server(self.root, self.backend, self.output)
        self.input = queue.Queue()
        self.thread = threading.Thread(target=self.server.run, args=(iter(self.input.get, None),), daemon=True)
        self.thread.start()

    def tearDown(self):
        self.input.put(None)
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive(), "Worker did not shut down")
        self.directory.cleanup()

    def command(self, **changes):
        request = dict(kind="music", prompt="Acoustic folk", music={"lyrics": "[Verse]\nA little song"},
                       duration=10, denoisingSteps=32, seed=42,
                       modelDirectory=self.root.as_uri(), outputURL=(self.root / (str(uuid.uuid4()) + ".wav")).as_uri())
        request.update(changes)
        return dict(protocolVersion=worker.PROTOCOL_VERSION, requestID=str(uuid.uuid4()),
                    jobID=str(uuid.uuid4()), kind="generate", generation=request)

    def send(self, command):
        self.input.put(json.dumps(command) + "\n")

    def terminal(self):
        while True:
            event = self.output.events.get(timeout=2)
            if event["kind"] in ("failed", "completed", "cancelled", "ready"):
                return event

    def test_preserves_each_step_count_and_accepts_immediate_next_job(self):
        for steps in (8, 16, 32):
            command = self.command(denoisingSteps=steps)
            self.send(command)
            event = self.terminal()
            self.assertEqual(event["kind"], "completed")
            self.assertEqual(event["jobID"], command["jobID"])
            self.assertEqual(event["outputDuration"], 0.75)
            self.assertEqual(self.backend.requests[-1]["steps"], steps)
            self.assertEqual(self.backend.requests[-1]["lyrics"], "[Verse]\nA little song")

    def test_default_steps(self):
        command = self.command()
        del command["generation"]["denoisingSteps"]
        self.send(command)
        self.assertEqual(self.terminal()["kind"], "completed")
        self.assertEqual(self.backend.requests[-1]["steps"], 32)

    def test_rejects_unsupported_steps_before_backend(self):
        for steps in (0, 1, 12, 64, True, 16.0):
            self.send(self.command(denoisingSteps=steps))
            self.assertEqual(self.terminal()["kind"], "failed")
        self.assertEqual(self.backend.requests, [])

    def test_rejects_missing_lyrics_and_invalid_length(self):
        for changes in ({"music": None}, {"music": {"lyrics": " "}},
                        {"duration": 0}, {"duration": 31}, {"duration": float("inf")},
                        {"kind": "soundEffect"}, {"blockCache": True}):
            self.send(self.command(**changes))
            self.assertEqual(self.terminal()["kind"], "failed")
        self.assertEqual(self.backend.requests, [])

    def test_rejects_other_package_remote_output_and_overwrite(self):
        existing = self.root / "existing.wav"
        existing.write_bytes(b"keep me")
        for changes in ({"modelDirectory": (self.root / "other").as_uri()},
                        {"outputURL": "https://example.com/audio.wav"},
                        {"outputURL": existing.as_uri()}):
            self.send(self.command(**changes))
            self.assertEqual(self.terminal()["kind"], "failed")
        self.assertEqual(existing.read_bytes(), b"keep me")

    def test_capabilities_and_version_rejection(self):
        command = dict(protocolVersion=worker.PROTOCOL_VERSION, requestID=str(uuid.uuid4()), kind="handshake")
        self.send(command)
        event = self.terminal()
        self.assertIn("musicGeneration", event["capabilities"]["features"])
        self.assertNotIn("videoGeneration", event["capabilities"]["features"])
        command["protocolVersion"] -= 1
        self.send(command)
        self.assertEqual(self.terminal()["kind"], "failed")

    def test_cancel_acknowledges_original_job_and_worker_is_reusable(self):
        self.backend.block = True
        command = self.command()
        self.send(command)
        self.assertEqual(self.output.events.get(timeout=2)["kind"], "accepted")
        self.send(dict(protocolVersion=worker.PROTOCOL_VERSION, requestID=str(uuid.uuid4()),
                       jobID=command["jobID"], kind="cancel"))
        event = self.terminal()
        self.assertEqual(event["kind"], "cancelled")
        self.assertEqual(event["requestID"], command["requestID"])
        self.assertFalse(worker.file_path(command["generation"]["outputURL"]).exists())
        self.backend.block = False
        self.send(self.command(denoisingSteps=8))
        self.assertEqual(self.terminal()["kind"], "completed")


if __name__ == "__main__":
    unittest.main()
