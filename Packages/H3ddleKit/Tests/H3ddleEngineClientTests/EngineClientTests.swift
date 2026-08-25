import Foundation
import H3ddleEngineProtocol
import H3ddleGeneration
import Testing

@testable import H3ddleEngineClient

@Suite("Engine process client")
struct EngineClientTests {
  @Test("Recovery manifests survive a fresh store and explicit discard removes them")
  func recoveryManifestRoundTrip() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "H3ddleRecoveryTests-\(UUID().uuidString)", isDirectory: true)
    let output = root.appendingPathComponent("outputs", isDirectory: true)
    let store = EngineGenerationRecoveryStore(rootURL: root, outputDirectory: output)
    let recovery = store.makeContext(for: .video)
    let request = GenerationRequest(
      kind: .video,
      prompt: "A lighthouse in rain",
      duration: 5,
      seed: 7,
      recovery: recovery
    )
    try store.save(
      EngineGenerationRecoveryRecord(
        request: request,
        modelDirectory: URL(fileURLWithPath: "/tmp/h3-model", isDirectory: true),
        displayPrompt: "A lighthouse in rain",
        settingsDescription: "video · 5s"
      )
    )

    let restored = try #require(store.latest())
    #expect(restored.request == request)
    #expect(restored.displayPrompt == "A lighthouse in rain")
    let manifest = recovery.directoryURL.appendingPathComponent(
      EngineGenerationRecoveryStore.manifestName)
    let attributes = try FileManager.default.attributesOfItem(atPath: manifest.path)
    #expect(attributes[.posixPermissions] as? Int == 0o600)

    store.discard(recovery)
    #expect(store.latest() == nil)
  }

  @Test("A checkpoint fingerprint is stable and changes with generation inputs")
  func checkpointFingerprintBindsInputs() throws {
    let model = FileManager.default.temporaryDirectory.appendingPathComponent(
      "H3ddleFingerprintModel-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: model) }
    let marker = model.appendingPathComponent("model.json")
    try Data("one".utf8).write(to: marker)
    let output = URL(fileURLWithPath: "/tmp/output.mp4")
    let first = EngineGenerationRequest(
      kind: .video,
      prompt: "one prompt",
      duration: 5,
      seed: 42,
      modelDirectory: model,
      outputURL: output
    )
    var changed = first
    changed.prompt = "another prompt"

    let fingerprint = try EngineGenerationProvider.checkpointFingerprint(
      for: first, environment: [:])
    let repeated = try EngineGenerationProvider.checkpointFingerprint(
      for: first, environment: [:])
    let changedPrompt = try EngineGenerationProvider.checkpointFingerprint(
      for: changed, environment: [:])
    let changedEnvironment = try EngineGenerationProvider.checkpointFingerprint(
      for: first, environment: ["H3_SOL_ATTN": "1"])
    #expect(fingerprint.count == 64)
    #expect(fingerprint == repeated)
    #expect(fingerprint != changedPrompt)
    #expect(fingerprint != changedEnvironment)
  }

  @Test("A recoverable H3 request carries its checkpoint across the helper protocol")
  func checkpointCrossesTheProtocol() async throws {
    let session = fakeEngineSession(arguments: ["--echo-generate"])
    defer { session.shutdown() }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "H3ddleCheckpointWire-\(UUID().uuidString)", isDirectory: true)
    let recovery = GenerationRecoveryContext(
      jobID: UUID(),
      directoryURL: root,
      outputURL: root.appendingPathComponent("output.mp4")
    )
    let provider = EngineGenerationProvider(
      session: session,
      modelDirectory: URL(fileURLWithPath: "/tmp/h3-package", isDirectory: true),
      outputDirectory: root
    )
    var sent: [String: Any]?
    for try await event in provider.events(
      for: GenerationRequest(
        kind: .video,
        prompt: "A recoverable clip",
        duration: 5,
        recovery: recovery
      )
    ) {
      if case .progress(let phase, _) = event,
        let data = phase.data(using: .utf8),
        let decoded = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
      {
        sent = decoded
      }
    }
    let generation = try #require(sent)
    let checkpoint = try #require(generation["checkpoint"] as? [String: Any])
    #expect((checkpoint["fileURL"] as? String)?.hasSuffix("/denoiser.h3ckpt") == true)
    #expect((checkpoint["fingerprint"] as? String)?.count == 64)
  }

  @Test("A recoverable job relaunches the helper once after an unexpected exit")
  func recoverableJobRestartsTheHelper() async throws {
    let marker = FileManager.default.temporaryDirectory.appendingPathComponent(
      "H3ddleCrashOnce-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: marker) }
    let session = fakeEngineSession(arguments: ["--crash-once-file", marker.path])
    defer { session.shutdown() }
    let recoveryRoot = FileManager.default.temporaryDirectory.appendingPathComponent(
      "H3ddleRestartRecovery-\(UUID().uuidString)", isDirectory: true)
    let recovery = GenerationRecoveryContext(
      jobID: UUID(),
      directoryURL: recoveryRoot,
      outputURL: recoveryRoot.appendingPathComponent("output.mp4")
    )
    let provider = EngineGenerationProvider(
      session: session,
      modelDirectory: URL(fileURLWithPath: "/tmp/h3-package", isDirectory: true)
    )
    var completed = false
    for try await event in provider.events(
      for: GenerationRequest(
        kind: .video,
        prompt: "Restart fixture",
        duration: 1,
        recovery: recovery
      )
    ) {
      if case .completed = event { completed = true }
    }
    #expect(completed)
    #expect(session.helperLaunchCount == 2)
  }

  @Test("A missing helper produces a useful error")
  func missingExecutable() async {
    let client = EngineProcessClient(
      executableURL: URL(fileURLWithPath: "/path/that/does/not/exist")
    )

    await #expect(throws: EngineClientError.self) {
      _ = try await client.capabilities()
    }
  }

  @Test("The bundled helper lives in the app Helpers directory")
  func bundledLocation() {
    let bundleURL = URL(fileURLWithPath: "/Applications/H3ddle.app", isDirectory: true)
    #expect(
      EngineExecutableLocator.bundled(at: bundleURL).path
        == "/Applications/H3ddle.app/Contents/Helpers/H3ddleEngineService"
    )
  }

  @Test("A missing helper fails the same way for audio as for video")
  func audioUsesTheNativeHelper() async {
    let provider = EngineGenerationProvider(
      executableURL: URL(fileURLWithPath: "/path/that/does/not/exist"),
      modelDirectory: URL(fileURLWithPath: "/tmp/model", isDirectory: true)
    )
    let request = GenerationRequest(kind: .audio, prompt: "Soft room tone", duration: 4)

    await #expect(throws: EngineClientError.self) {
      for try await _ in provider.events(for: request) {}
    }
  }

  @Test("Audio results are written as WAV, so no mux or extraction is involved")
  func audioResultsAreWAV() {
    #expect(EngineGenerationProvider.outputExtension(for: .audio) == "wav")
    #expect(EngineGenerationProvider.outputExtension(for: .video) == "mp4")
    #expect(EngineGenerationProvider.outputExtension(for: .image) == "png")
  }

  @Test("A helper process is reused across handshake and inspection")
  func sessionReusesHelperProcess() async throws {
    let session = fakeEngineSession()
    defer { session.shutdown() }

    _ = try await session.capabilities()
    let first = session.processIdentifier
    #expect(first != nil)

    let report = try await session.inspectModel(
      at: URL(fileURLWithPath: "/tmp/model", isDirectory: true)
    )
    #expect(report.supportsGeneration)
    #expect(session.processIdentifier == first)
  }

  @Test("Completing a generation does not cancel the shared session")
  func completedGenerationLeavesSessionReadyForTheNextJob() async throws {
    let session = fakeEngineSession(arguments: ["--exit-on-idle-cancel"])
    defer { session.shutdown() }
    let provider = EngineGenerationProvider(
      session: session,
      modelDirectory: URL(fileURLWithPath: "/tmp/model", isDirectory: true)
    )

    for prompt in ["First queued job", "Second queued job"] {
      var completed = false
      for try await event in provider.events(
        for: GenerationRequest(kind: .video, prompt: prompt, duration: 1)
      ) {
        if case .completed = event { completed = true }
      }
      #expect(completed)
    }

    #expect(session.helperLaunchCount == 1)
  }

  @Test("Cancelling generation keeps the helper process alive")
  func cancellationKeepsHelperAlive() async throws {
    let session = fakeEngineSession(arguments: ["--hold-generate"])
    defer { session.shutdown() }

    let provider = EngineGenerationProvider(
      session: session,
      modelDirectory: URL(fileURLWithPath: "/tmp/model", isDirectory: true)
    )
    let request = GenerationRequest(
      kind: .video,
      prompt: "Cancellation fixture",
      duration: 1
    )
    _ = try await session.capabilities()
    let first = session.processIdentifier
    #expect(first != nil)

    let generation = Task {
      do {
        for try await _ in provider.events(for: request) {}
      } catch {
        // Cooperative cancel should finish the stream; a dying fixture
        // must not fail the reuse assertion below.
      }
    }

    try await Task.sleep(for: .milliseconds(150))
    generation.cancel()
    await generation.value
    #expect(session.processIdentifier == first)
  }

  @Test("Inspection fails immediately while a generation is running")
  func inspectFailsFastWhenBusy() async throws {
    let session = fakeEngineSession(arguments: ["--hold-generate"])
    defer { session.shutdown() }

    _ = try await session.capabilities()
    let provider = EngineGenerationProvider(
      session: session,
      modelDirectory: URL(fileURLWithPath: "/tmp/model", isDirectory: true)
    )
    let generation = Task {
      do {
        for try await _ in provider.events(
          for: GenerationRequest(kind: .video, prompt: "Busy inspect", duration: 1)
        ) {}
      } catch {}
    }

    try await Task.sleep(for: .milliseconds(150))
    await #expect(throws: EngineClientError.busy) {
      _ = try await session.inspectModel(
        at: URL(fileURLWithPath: "/tmp/model", isDirectory: true)
      )
    }
    generation.cancel()
    await generation.value
  }

  @Test("A cancel that is not acknowledged kills the helper so the next job can start")
  func unacknowledgedCancelRespawnsHelper() async throws {
    let session = fakeEngineSession(arguments: ["--hold-generate", "--ignore-cancel"])
    session.cancelEscalation = .milliseconds(80)
    defer { session.shutdown() }

    _ = try await session.capabilities()
    #expect(session.processIdentifier != nil)
    let launchesBefore = session.helperLaunchCount

    let provider = EngineGenerationProvider(
      session: session,
      modelDirectory: URL(fileURLWithPath: "/tmp/model", isDirectory: true)
    )
    // The fake helper emits one progress event and then holds; cancelling on
    // that event guarantees the job is in flight at any machine speed, where
    // a timed sleep raced slow runners.
    let (inFlight, inFlightSignal) = AsyncStream.makeStream(of: Void.self)
    let generation = Task {
      do {
        for try await _ in provider.events(
          for: GenerationRequest(kind: .video, prompt: "Hard cancel", duration: 1)
        ) {
          inFlightSignal.yield(())
        }
      } catch {}
      inFlightSignal.finish()
    }

    var signals = inFlight.makeAsyncIterator()
    _ = await signals.next()
    generation.cancel()
    await generation.value
    // The ignored cancel escalates to SIGKILL on an internal timer; wait for
    // the process to actually die rather than sleeping past the timer.
    for _ in 0..<200 where session.processIdentifier != nil {
      try await Task.sleep(for: .milliseconds(20))
    }
    #expect(session.processIdentifier == nil)

    _ = try await session.capabilities()
    #expect(session.processIdentifier != nil)
    // The OS may recycle the killed helper's pid for its replacement — CI
    // runners do — so the fresh launch is the assertion, not a changed pid.
    #expect(session.helperLaunchCount == launchesBefore + 1)
  }

  @Test("Helper stderr is drained so a long session does not stall")
  func stderrIsDrained() async throws {
    let session = fakeEngineSession(arguments: ["--spam-stderr"])
    defer { session.shutdown() }

    _ = try await session.capabilities()
    let report = try await session.inspectModel(
      at: URL(fileURLWithPath: "/tmp/model", isDirectory: true)
    )
    #expect(report.supportsGeneration)
  }

  /// The seam between the app's request and the engine's, which no other test
  /// crosses: a speech job is the only kind whose settings do not survive as
  /// defaults if a field is dropped, because there is no voice without them.
  @Test("A speech request crosses the protocol as a speech job with its voice")
  func speechRequestCrossesTheProtocol() async throws {
    let session = fakeEngineSession(arguments: ["--echo-generate"])
    defer { session.shutdown() }
    let provider = EngineGenerationProvider(
      session: session,
      modelDirectory: URL(fileURLWithPath: "/tmp/speech-package", isDirectory: true)
    )
    let request = GenerationRequest(
      kind: .audio,
      audioEngine: .speech,
      speech: EngineSpeechOptions(
        referenceAudioURL: URL(fileURLWithPath: "/tmp/voice.wav"),
        language: .japanese,
        temperature: 0.75,
        topK: 40,
        repetitionPenalty: 1.1
      ),
      prompt: "Read this aloud.",
      duration: 12
    )

    var sent: [String: Any]?
    for try await event in provider.events(for: request) {
      if case .progress(let phase, _) = event,
        let data = phase.data(using: .utf8),
        let decoded = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
      {
        sent = decoded
      }
    }
    let generation = try #require(sent)
    #expect(generation["kind"] as? String == "speech")
    let speech = try #require(generation["speech"] as? [String: Any])
    #expect(speech["language"] as? String == "ja")
    #expect(speech["temperature"] as? Double == 0.75)
    #expect(speech["topK"] as? Int == 40)
    #expect(speech["repetitionPenalty"] as? Double == 1.1)
    #expect((speech["referenceAudioURL"] as? String)?.hasSuffix("/tmp/voice.wav") == true)
  }

  @Test("An H3 inpainting request crosses the protocol with source and mask")
  func inpaintingRequestCrossesTheProtocol() async throws {
    let session = fakeEngineSession(arguments: ["--echo-generate"])
    defer { session.shutdown() }
    let provider = EngineGenerationProvider(
      session: session,
      modelDirectory: URL(fileURLWithPath: "/tmp/h3-package", isDirectory: true)
    )
    let request = GenerationRequest(
      kind: .video,
      videoEngine: .h3,
      videoInpainting: EngineVideoInpaintingOptions(
        sourceVideoURL: URL(fileURLWithPath: "/tmp/source.mov"),
        maskURL: URL(fileURLWithPath: "/tmp/mask.png"),
        preserveSourceAudio: true
      ),
      prompt: "replace the white mask with flowers",
      duration: 5,
      referenceImageURLs: [URL(fileURLWithPath: "/tmp/flowers.png")]
    )

    var sent: [String: Any]?
    for try await event in provider.events(for: request) {
      if case .progress(let phase, _) = event,
        let data = phase.data(using: .utf8),
        let decoded = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
      {
        sent = decoded
      }
    }

    let generation = try #require(sent)
    let video = try #require(generation["video"] as? [String: Any])
    let inpainting = try #require(video["inpainting"] as? [String: Any])
    #expect(video["model"] as? String == "h3")
    #expect((inpainting["sourceVideoURL"] as? String)?.hasSuffix("/tmp/source.mov") == true)
    #expect((inpainting["maskURL"] as? String)?.hasSuffix("/tmp/mask.png") == true)
    #expect(inpainting["maskKind"] as? String == "still")
    #expect(inpainting["preserveSourceAudio"] as? Bool == true)
  }

  /// The neutral voice: no clip, and the options still have to travel. An
  /// empty `speech` block would be a request the engine refuses outright, so
  /// "no reference" must not collapse into "no speech settings".
  @Test("Speech with no reference clip still carries its speech settings")
  func speechWithoutAReferenceStillCarriesSettings() async throws {
    let session = fakeEngineSession(arguments: ["--echo-generate"])
    defer { session.shutdown() }
    let provider = EngineGenerationProvider(
      session: session,
      modelDirectory: URL(fileURLWithPath: "/tmp/speech-package", isDirectory: true)
    )
    var sent: [String: Any]?
    for try await event in provider.events(
      for: GenerationRequest(
        kind: .audio,
        audioEngine: .speech,
        speech: EngineSpeechOptions(language: .german, temperature: 0.8),
        prompt: "Sprich diesen Satz.",
        duration: 8
      )
    ) {
      if case .progress(let phase, _) = event,
        let data = phase.data(using: .utf8),
        let decoded = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
      {
        sent = decoded
      }
    }
    let generation = try #require(sent)
    #expect(generation["kind"] as? String == "speech")
    let speech = try #require(generation["speech"] as? [String: Any])
    #expect(speech["language"] as? String == "de")
    #expect(speech["referenceAudioURL"] == nil)
    #expect(speech["voiceEmbeddingURL"] == nil)
  }

  /// Music and sound effects still reach the engine as sound-effect jobs, and
  /// H3's own soundtrack still reaches it as audio. Replacing a boolean with
  /// an enum is exactly the change that quietly reroutes one of them.
  @Test("The other audio engines still map where they did")
  func otherAudioEnginesAreUnchanged() async throws {
    for (engine, expected) in [
      (AudioGenerationEngine.h3, "audio"),
      (AudioGenerationEngine.stableAudio, "soundEffect"),
    ] {
      let session = fakeEngineSession(arguments: ["--echo-generate"])
      defer { session.shutdown() }
      let provider = EngineGenerationProvider(
        session: session,
        modelDirectory: URL(fileURLWithPath: "/tmp/model", isDirectory: true)
      )
      var sent: [String: Any]?
      for try await event in provider.events(
        for: GenerationRequest(
          kind: .audio, audioEngine: engine, prompt: "Rain on a tin roof", duration: 4
        )
      ) {
        if case .progress(let phase, _) = event,
          let data = phase.data(using: .utf8),
          let decoded = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
          sent = decoded
        }
      }
      #expect(try #require(sent)["kind"] as? String == expected)
    }
  }

  /// Confirmation is an app decision, but the engine owns the final guard.
  /// The consent bit therefore has to survive both request representations
  /// rather than stopping at the Generation Studio button.
  @Test("LTX memory-overcommit consent crosses into the engine request")
  func ltxMemoryConsentCrossesTheProtocol() async throws {
    let session = fakeEngineSession(arguments: ["--echo-generate"])
    defer { session.shutdown() }
    let provider = EngineGenerationProvider(
      session: session,
      modelDirectory: URL(fileURLWithPath: "/tmp/ltx-package", isDirectory: true)
    )
    var sent: [String: Any]?
    for try await event in provider.events(
      for: GenerationRequest(
        kind: .video,
        videoEngine: .ltx,
        prompt: "A long coastal tracking shot",
        duration: 10,
        canvasWidth: 1248,
        canvasHeight: 704,
        allowsLTXMemoryOvercommit: true
      )
    ) {
      if case .progress(let phase, _) = event,
        let data = phase.data(using: .utf8),
        let decoded = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
      {
        sent = decoded
      }
    }

    let generation = try #require(sent)
    #expect(generation["video"] as? [String: String] == ["model": "ltx"])
    #expect(generation["allowsLTXMemoryOvercommit"] as? Bool == true)
  }
}

private func fakeEngineSession(arguments: [String] = []) -> EngineSession {
  let script = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("Fixtures/fake-engine.py")
  return EngineSession(
    executableURL: URL(fileURLWithPath: "/usr/bin/python3"),
    arguments: [script.path] + arguments
  )
}
