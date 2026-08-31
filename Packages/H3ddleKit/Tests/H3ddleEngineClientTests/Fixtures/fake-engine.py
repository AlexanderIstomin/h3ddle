#!/usr/bin/env python3
import json
import os
import sys
from urllib.parse import unquote, urlparse

HOLD_GENERATE = "--hold-generate" in sys.argv
IGNORE_CANCEL = "--ignore-cancel" in sys.argv
EXIT_ON_IDLE_CANCEL = "--exit-on-idle-cancel" in sys.argv
SPAM_STDERR = "--spam-stderr" in sys.argv
# Reports the generation block exactly as it arrived, so a test can assert
# what crossed the protocol rather than what the caller meant to send.
ECHO_GENERATE = "--echo-generate" in sys.argv
OMIT_OUTPUT = "--omit-output" in sys.argv
CRASH_ONCE_FILE = None
if "--crash-once-file" in sys.argv:
    marker = sys.argv.index("--crash-once-file")
    if marker + 1 < len(sys.argv):
        CRASH_ONCE_FILE = sys.argv[marker + 1]
active_generate = None

CAPABILITIES = {
    "engineName": "fake-h3",
    "engineVersion": "0",
    "features": [
        "modelInspection",
        "videoGeneration",
        "imageGeneration",
        "standaloneAudioGeneration",
        "embeddedAudio",
        "cancellation",
        "denoisingPreviews",
        "videoInpainting",
    ],
}


def emit(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def base(command, kind):
    return {
        "protocolVersion": command.get("protocolVersion", 8),
        "requestID": command.get("requestID"),
        "jobID": command.get("jobID"),
        "kind": kind,
    }


for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    command = json.loads(line)
    kind = command.get("kind")
    if kind == "handshake":
        if SPAM_STDERR:
            sys.stderr.write("x" * (80 * 1024) + "\n")
            sys.stderr.flush()
        event = base(command, "ready")
        event["capabilities"] = CAPABILITIES
        event["message"] = "H3 native engine ready"
        emit(event)
    elif kind == "inspectModel":
        inspection = command.get("modelInspection") or {}
        event = base(command, "modelInspected")
        event["model"] = {
            "modelDirectory": inspection.get("modelDirectory", "file:///tmp/model"),
            "components": [],
            "device": {
                "name": "Fake GPU",
                "architecture": "test",
                "physicalMemory": 1,
                "recommendedWorkingSet": 1,
                "unifiedMemory": True,
            },
            "format": "optimizedINT8SingleFile",
            "supportsGeneration": True,
        }
        emit(event)
    elif kind == "generate":
        active_generate = command
        accepted = base(command, "accepted")
        accepted["capabilities"] = CAPABILITIES
        emit(accepted)
        if CRASH_ONCE_FILE:
            try:
                with open(CRASH_ONCE_FILE, "x", encoding="utf-8") as marker_file:
                    marker_file.write("crashed")
                sys.exit(73)
            except FileExistsError:
                pass
        if HOLD_GENERATE:
            # One progress event before holding, so tests can synchronize on
            # the job being in flight instead of sleeping and hoping.
            progress = base(command, "progress")
            progress["phase"] = "holding"
            progress["fractionComplete"] = 0
            emit(progress)
            continue
        generation = command.get("generation") or {}
        if ECHO_GENERATE:
            echo = base(command, "progress")
            echo["phase"] = json.dumps(generation, sort_keys=True)
            echo["fractionComplete"] = 0
            emit(echo)
        completed = base(command, "completed")
        output_url = generation.get("outputURL", "file:///tmp/out.mp4")
        if not OMIT_OUTPUT:
            parsed = urlparse(output_url)
            if parsed.scheme == "file":
                output_path = unquote(parsed.path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as output_file:
                    output_file.write(b"fake engine output")
        completed["outputURL"] = output_url
        completed["outputDuration"] = 1
        emit(completed)
        active_generate = None
    elif kind == "cancel":
        if EXIT_ON_IDLE_CANCEL and active_generate is None:
            sys.exit(74)
        if IGNORE_CANCEL:
            continue
        target = active_generate or command
        emit(base(target, "cancelled"))
        active_generate = None
    elif kind == "shutdown":
        break
