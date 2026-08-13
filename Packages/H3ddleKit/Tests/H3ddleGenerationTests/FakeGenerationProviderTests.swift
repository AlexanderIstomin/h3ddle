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

    for try await event in provider.events(for: request) {
      switch event {
      case .progress:
        progressCount += 1
      case .completed(let asset):
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

  @Test("An empty prompt fails visibly")
  func rejectsEmptyPrompt() async {
    let provider = FakeGenerationProvider(stepDelay: .zero)
    let request = GenerationRequest(kind: .video, prompt: "   ", duration: 5)

    await #expect(throws: GenerationError.emptyPrompt) {
      for try await _ in provider.events(for: request) {}
    }
  }
}
