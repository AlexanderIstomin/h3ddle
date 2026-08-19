import Foundation
import H3ddleEngineProtocol
import H3ddleGeneration
import Testing

@testable import H3ddleEngineClient

@Suite("Engine process client")
struct EngineClientTests {
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
