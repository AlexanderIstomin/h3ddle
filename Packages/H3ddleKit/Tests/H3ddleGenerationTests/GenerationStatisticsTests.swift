import Foundation
import Testing

@testable import H3ddleGeneration

@Suite("Generation statistics")
struct GenerationStatisticsTests {
  private func statistics(
    kind: GenerationKind = .image,
    seconds: TimeInterval = 361,
    blockCache: Bool = false
  ) -> GenerationStatistics {
    GenerationStatistics(
      kind: kind,
      seconds: seconds,
      canvasWidth: 448,
      canvasHeight: 448,
      denoisingSteps: 8,
      clipSeconds: 3,
      modelName: "MiniMax H3 · Turbo",
      blockCache: blockCache,
      deviceName: "Apple M1 Pro",
      deviceMemoryBytes: 34_359_738_368
    )
  }

  @Test("A still reads as a sentence with machine, time, and settings")
  func imageSummary() {
    let summary = statistics().socialSummary
    #expect(summary.hasPrefix("Generated an image at 448x448 in 6 min 1 s"))
    #expect(summary.contains("on Apple M1 Pro with 32 GB, fully offline."))
    #expect(summary.contains("Settings: 8 denoising passes, model MiniMax H3 · Turbo."))
    #expect(summary.contains("github.com/AlexanderIstomin/h3ddle"))
  }

  @Test("Video summaries name the clip length and its sound")
  func videoSummary() {
    let summary = statistics(kind: .video).socialSummary
    #expect(summary.hasPrefix("Generated a 3.0s video with sound at 448x448"))
  }

  @Test("Audio summaries carry no canvas")
  func audioSummary() {
    var stats = statistics(kind: .audio)
    stats.canvasWidth = nil
    stats.canvasHeight = nil
    #expect(stats.socialSummary.hasPrefix("Generated 3.0s of audio in 6 min 1 s"))
    #expect(!stats.socialSummary.contains(" at "))
  }

  @Test("The block cache is mentioned only when it ran")
  func blockCacheMention() {
    #expect(!statistics().socialSummary.contains("block cache"))
    #expect(statistics(blockCache: true).socialSummary.contains("block cache on"))
  }

  @Test("Sub-minute and whole-minute runs read naturally")
  func elapsedPhrasing() {
    #expect(statistics(seconds: 48).elapsedPhrase == "48 s")
    #expect(statistics(seconds: 120).elapsedPhrase == "2 min")
    #expect(statistics(seconds: 3_601).elapsedPhrase == "60 min 1 s")
  }

  @Test("An unknown machine still produces a usable sentence")
  func unknownDevice() {
    var stats = statistics()
    stats.deviceName = nil
    stats.deviceMemoryBytes = nil
    #expect(stats.socialSummary.contains("on Apple silicon, fully offline."))
  }
}
