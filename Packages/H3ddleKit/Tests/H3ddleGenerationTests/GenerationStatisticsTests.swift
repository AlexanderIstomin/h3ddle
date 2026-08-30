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
      aspectRatio: "1:1",
      transformerBlocks: 45,
      coreReuse: 4,
      blockCache: blockCache,
      stillFrameCount: 22,
      previewDenoise: true,
      seed: 12_345,
      conditioning: "none",
      deviceName: "Apple M1 Pro",
      deviceMemoryBytes: 34_359_738_368
    )
  }

  @Test("A still reads as a sentence with machine, time, and settings")
  func imageSummary() {
    let summary = statistics().socialSummary
    #expect(summary.hasPrefix("Generated an image at 448x448 in 6 min 1 s"))
    #expect(summary.contains("on Apple M1 Pro with 32 GB, fully offline."))
    #expect(summary.contains("Settings: model MiniMax H3 · Turbo, aspect 1:1, 8 passes"))
    #expect(summary.contains("45 transformer blocks, core reuse every 4th pass"))
    #expect(summary.contains("still detail Full 22f, denoising preview on"))
    #expect(summary.contains("seed 12345, conditioning none"))
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
    stats.aspectRatio = nil
    stats.transformerBlocks = nil
    stats.coreReuse = nil
    stats.blockCache = nil
    stats.stillFrameCount = nil
    stats.previewDenoise = nil
    stats.seed = nil
    stats.conditioning = nil
    #expect(stats.socialSummary.hasPrefix("Generated 3.0s of audio in 6 min 1 s"))
    #expect(!stats.socialSummary.contains(" at "))
  }

  @Test("Repeatable switches state whether they were on or off")
  func blockCacheMention() {
    #expect(statistics().socialSummary.contains("block cache off"))
    #expect(statistics(blockCache: true).socialSummary.contains("block cache on"))
  }

  @Test("Model-specific step wording and speech settings stay user-facing")
  func modelSpecificSettings() {
    var ltx = statistics(kind: .video)
    ltx.stepLabel = "steps"
    ltx.transformerBlocks = nil
    ltx.coreReuse = nil
    ltx.blockCache = nil
    ltx.stillFrameCount = nil
    ltx.previewDenoise = nil
    #expect(ltx.socialSummary.contains("8 steps"))
    #expect(!ltx.socialSummary.contains("transformer blocks"))

    var speech = statistics(kind: .audio)
    speech.canvasWidth = nil
    speech.canvasHeight = nil
    speech.denoisingSteps = nil
    speech.aspectRatio = nil
    speech.transformerBlocks = nil
    speech.coreReuse = nil
    speech.blockCache = nil
    speech.stillFrameCount = nil
    speech.previewDenoise = nil
    speech.seed = nil
    speech.conditioning = nil
    speech.speechLanguage = "English"
    speech.speechVariation = 0.9
    speech.voiceName = "Neutral"
    #expect(!speech.socialSummary.contains("passes"))
    #expect(speech.socialSummary.contains("language English, variation 0.90, voice Neutral"))
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

  @Test("Phase timing and sampled peak memory survive in the copied summary")
  func performanceTelemetry() {
    var stats = statistics(kind: .video)
    stats.phaseDurations = [
      GenerationPhaseTimeline.Entry(phase: "Preparing generation", duration: 12.25),
      GenerationPhaseTimeline.Entry(phase: "denoise", duration: 240.5),
    ]
    stats.peakEngineMemoryBytes = 21_474_836_480

    #expect(
      stats.socialSummary.contains(
        "Performance: Preparing generation 12.2s · denoise 240.5s; "
          + "peak sampled engine memory 20.0 GB."
      )
    )
  }
}
