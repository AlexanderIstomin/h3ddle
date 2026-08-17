import Foundation
import Testing

@testable import H3ddleGeneration

@Suite("Remaining time")
struct GenerationRemainingTests {
  @Test("Phrases stay short enough to sit inside the progress ring")
  func phrasing() {
    #expect(GenerationRemaining.phrase(20) == "~1 min")
    #expect(GenerationRemaining.phrase(70) == "~1 min")
    #expect(GenerationRemaining.phrase(400) == "~7 min")
    #expect(GenerationRemaining.phrase(3_600) == "~1 h")
    #expect(GenerationRemaining.phrase(9_000) == "~2.5 h")
    for seconds in [30.0, 100.0, 4_000.0, 20_000.0] {
      #expect(GenerationRemaining.phrase(seconds).count <= 7)
    }
  }

  @Test("A finished or nonsensical estimate never reads as a countdown")
  func degenerate() {
    #expect(GenerationRemaining.phrase(0) == "almost done")
    #expect(GenerationRemaining.phrase(-5) == "almost done")
    #expect(GenerationRemaining.phrase(.infinity) == "almost done")
  }

  @Test("Denoising spans the run, not the current step")
  func denoiseSpansTheRun() {
    // Halfway through the first of six steps is a twelfth of the denoising.
    let early = GenerationRemaining.denoiseFraction(
      phase: "denoise step 1/6 transformer", phaseFraction: 0.5)
    #expect(early == 0.5 / 6)
    // The same half-done phase reads far later when the step is later.
    let late = GenerationRemaining.denoiseFraction(
      phase: "denoise step 5/6 transformer", phaseFraction: 0.5)
    #expect(late == 4.5 / 6)
  }

  /// The bar leaves room at the bottom for loading and encoding, so the two
  /// minutes before the first pass are not a stationary bar. The mapping has
  /// to be monotonic across the join: preparation ends at the share, and the
  /// first pass must not land below it.
  @Test("Denoising is mapped above the preparation band, without going back")
  func overallLeavesRoomForPreparation() {
    let share = GenerationRemaining.preparationShare
    let firstPass = GenerationRemaining.overallProgress(
      phase: "denoise step 1/6 transformer", phaseFraction: 0)
    #expect(firstPass == share)
    #expect(GenerationRemaining.preparationProgress(phaseFraction: 1) == share)

    // The last pass leaves room for the decoder rather than reaching 100%:
    // the picture is not finished when the denoising is.
    let done = GenerationRemaining.overallProgress(
      phase: "denoise step 6/6 transformer", phaseFraction: 1)
    #expect(done == GenerationRemaining.decodeProgress)
    #expect(GenerationRemaining.decodeProgress < 1)

    // Preparation never reaches into the denoising band, whatever it reports.
    #expect(GenerationRemaining.preparationProgress(phaseFraction: 5) == share)
    #expect(GenerationRemaining.preparationProgress(phaseFraction: -1) == 0)
  }

  @Test("Phases without a step counter yield no overall position")
  func nonDenoisePhases() {
    #expect(GenerationRemaining.overallProgress(
      phase: "text encoder", phaseFraction: 0.5) == nil)
    #expect(GenerationRemaining.overallProgress(
      phase: "video VAE load", phaseFraction: 0.9) == nil)
  }

  @Test("Step counters are read out of the engine's phase names")
  func stepParsing() {
    let parsed = GenerationRemaining.denoiseStep(in: "denoise step 3/20 transformer")
    #expect(parsed?.step == 3)
    #expect(parsed?.total == 20)
    #expect(GenerationRemaining.denoiseStep(in: "denoise") == nil)
  }

  @Test("Estimates project from the pace measured so far")
  func projection() {
    #expect(GenerationRemaining.estimate(elapsed: 60, progress: 0.25) == 180)
    #expect(GenerationRemaining.estimate(elapsed: 60, progress: 0.01) == nil)
    #expect(GenerationRemaining.estimate(elapsed: 0.5, progress: 0.5) == nil)
    #expect(GenerationRemaining.estimate(elapsed: 60, progress: 1) == nil)
  }
}
