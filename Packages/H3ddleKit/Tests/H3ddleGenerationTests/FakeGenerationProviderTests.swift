import Foundation
import Testing

@testable import H3ddleGeneration

@Suite("Fake generation provider")
struct FakeGenerationProviderTests {
  @Test("Audio generation emits progress and a typed audio asset")
  func generatesAudio() async throws {
    let provider = FakeGenerationProvider(stepDelay: .zero)
    let request = GenerationRequest(
      kind: .audio,
      prompt: "Sparse room tone",
      duration: 8
    )
    var progressCount = 0
    var completedKind: GenerationKind?
    var outputURL: URL?
    defer { if let outputURL { try? FileManager.default.removeItem(at: outputURL) } }

    for try await event in provider.events(for: request) {
      switch event {
      case .progress:
        progressCount += 1
      case .resourceUsage:
        break
      case .completed(let asset):
        outputURL = asset.url
        completedKind = asset.kind == .audio ? .audio : nil
        #expect(FileManager.default.fileExists(atPath: asset.url.path))
        #expect(asset.url.pathExtension == "wav")
      case .preview:
        break
      }
    }

    #expect(progressCount == 3)
    #expect(completedKind == .audio)
  }

  @Test("Visual fakes return concrete non-empty artifacts")
  func visualFakesWriteArtifacts() async throws {
    let provider = FakeGenerationProvider(stepDelay: .zero)
    for kind in [GenerationKind.video, .image] {
      var outputURL: URL?
      for try await event in provider.events(
        for: GenerationRequest(kind: kind, prompt: "Fixture", duration: 1)
      ) {
        guard case .completed(let asset) = event else { continue }
        outputURL = asset.url
        let size = try asset.url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        #expect(size > 0)
      }
      let completedURL = try #require(outputURL)
      try? FileManager.default.removeItem(at: completedURL)
    }
  }

  @Test("A production fallback refuses to fake a missing model")
  func missingModelFailsClosed() async {
    let provider = MissingModelGenerationProvider()
    let request = GenerationRequest(kind: .video, prompt: "A real request", duration: 5)

    await #expect(throws: GenerationError.modelRequired) {
      for try await _ in provider.events(for: request) {}
    }
  }

  @Test("An empty prompt fails visibly")
  func rejectsEmptyPrompt() async {
    let provider = FakeGenerationProvider(stepDelay: .zero)
    let request = GenerationRequest(kind: .video, prompt: "   ", duration: 5)

    await #expect(throws: GenerationError.emptyPrompt) {
      for try await _ in provider.events(for: request) {}
    }
  }
}
