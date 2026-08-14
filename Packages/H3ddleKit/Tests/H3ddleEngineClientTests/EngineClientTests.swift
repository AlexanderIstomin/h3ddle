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
    let generation = Task {
      do {
        for try await _ in provider.events(
          for: GenerationRequest(kind: .video, prompt: "Hard cancel", duration: 1)
        ) {}
      } catch {}
    }

    try await Task.sleep(for: .milliseconds(150))
    generation.cancel()
    try await Task.sleep(for: .milliseconds(200))
    await generation.value

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
