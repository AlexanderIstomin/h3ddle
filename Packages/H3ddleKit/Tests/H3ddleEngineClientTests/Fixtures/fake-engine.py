#!/usr/bin/env python3
import json
import sys

HOLD_GENERATE = "--hold-generate" in sys.argv
IGNORE_CANCEL = "--ignore-cancel" in sys.argv
SPAM_STDERR = "--spam-stderr" in sys.argv
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
        if HOLD_GENERATE:
            # One progress event before holding, so tests can synchronize on
            # the job being in flight instead of sleeping and hoping.
            progress = base(command, "progress")
            progress["phase"] = "holding"
            progress["fractionComplete"] = 0
            emit(progress)
            continue
        generation = command.get("generation") or {}
        completed = base(command, "completed")
        completed["outputURL"] = generation.get("outputURL", "file:///tmp/out.mp4")
        completed["outputDuration"] = 1
        emit(completed)
        active_generate = None
    elif kind == "cancel":
        if IGNORE_CANCEL:
            continue
        target = active_generate or command
        emit(base(target, "cancelled"))
        active_generate = None
    elif kind == "shutdown":
        break
