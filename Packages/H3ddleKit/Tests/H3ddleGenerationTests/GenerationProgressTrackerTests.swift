import Foundation
import Testing

@testable import H3ddleGeneration

@Suite("Generation progress tracker")
struct GenerationProgressTrackerTests {
  /// The exact phase sequence a Z-Image render emits, taken from a recorded
  /// 512-pixel run: 35 encoder layers, the transformer, eight passes, the
  /// decoder. Fractions are phase-local, which is the whole difficulty —
  /// every pass reports 1.0, and the decoder reports 0.
  static func zImageEvents() -> [(phase: String, fraction: Double)] {
    var events: [(String, Double)] = []
    for layer in 0...35 {
      events.append(("text encoder", Double(layer) / 35))
    }
    events.append(("transformer", 0))
    events.append(("transformer", 1))
    for step in 1...8 {
      events.append(("denoise step \(step)/8", 1))
    }
    events.append(("image VAE", 0))
    return events
  }

  /// The bug this type exists to prevent: the bar filling up on every pass
  /// and dropping back to nothing at the next stage.
  @Test("A whole run advances and never goes backwards")
  func neverGoesBackwards() {
    var tracker = GenerationProgressTracker()
    var last = 0.0
    var elapsed = 0.0
    for event in Self.zImageEvents() {
      elapsed += 1
      tracker.record(
        phase: event.phase, phaseFraction: event.fraction, elapsed: elapsed)
      #expect(
        tracker.overall >= last,
        "\(event.phase) went back from \(last) to \(tracker.overall)"
      )
      #expect(tracker.overall <= 1)
      last = tracker.overall
    }
    // Ends on the decoder, with the passes done but the picture not yet out.
    #expect(tracker.overall == GenerationRemaining.decodeProgress)
    #expect(tracker.isFinishing)
  }

  @Test("Each stage lands in its own band")
  func stagesOccupyBands() {
    var tracker = GenerationProgressTracker()
    var elapsed = 0.0
    var atEndOfEncoder = 0.0
    var atFirstPass = 0.0
    for event in Self.zImageEvents() {
      elapsed += 1
      tracker.record(
        phase: event.phase, phaseFraction: event.fraction, elapsed: elapsed)
      if event.phase == "text encoder", event.fraction == 1 {
        atEndOfEncoder = tracker.overall
      }
      if event.phase == "denoise step 1/8" { atFirstPass = tracker.overall }
    }
    // Preparation fills its band and stops there.
    #expect(atEndOfEncoder == GenerationRemaining.preparationShare)
    // The first pass is above it — visibly, not by a rounding error.
    #expect(atFirstPass > GenerationRemaining.preparationShare + 0.05)
  }

  /// The estimate charged loading time against the passes and reported
  /// fifteen minutes for a render that took seventy-eight seconds.
  @Test("The estimate measures the passes, not the loading before them")
  func estimateExcludesPreparation() {
    var tracker = GenerationProgressTracker()
    // A minute of preparation, as a 512-pixel render really takes.
    for layer in 0...35 {
      tracker.record(
        phase: "text encoder",
        phaseFraction: Double(layer) / 35,
        elapsed: Double(layer) * 60 / 35
      )
    }
    #expect(tracker.remaining(elapsed: 60) == nil, "nothing to project from yet")

    // Then two seconds a pass. After the first, seven remain: ~14 seconds.
    tracker.record(phase: "denoise step 1/8", phaseFraction: 1, elapsed: 62)
    let remaining = tracker.remaining(elapsed: 62)
    #expect(remaining != nil)
    #expect(remaining! > 10 && remaining! < 20, "got \(remaining ?? -1)")
    // Emphatically not the whole-run projection, which is what went wrong.
    #expect(remaining! < 60)
  }

  @Test("LTX counts down its projection before measured timing takes over")
  func ltxProjectionHandoff() {
    var tracker = GenerationProgressTracker(profile: .ltx)
    tracker.record(phase: "text encoder", phaseFraction: 0.5, elapsed: 10)
    #expect(tracker.remaining(elapsed: 10, projectedTotal: 120) == 110)

    tracker.record(phase: "text encoder", phaseFraction: 1, elapsed: 20)
    tracker.record(phase: "denoise step 1/8", phaseFraction: 1, elapsed: 30)
    // Ten measured seconds for one eighth means 70 seconds of sampler work
    // remain. The settings projection retains 17% (20.4 seconds) for decode.
    let corrected = tracker.remaining(elapsed: 30, projectedTotal: 120)
    #expect(corrected != nil)
    #expect(abs(corrected! - 90.4) < 1e-9)
  }

  @Test("Standard and single-phase models also replace their initial projection")
  func otherModelProjectionHandoffs() {
    var standard = GenerationProgressTracker(profile: .standard)
    standard.record(phase: "text encoder", phaseFraction: 1, elapsed: 10)
    #expect(standard.remaining(elapsed: 10, projectedTotal: 120) == 110)
    standard.record(phase: "denoise step 1/8", phaseFraction: 1, elapsed: 20)
    // The measured seven remaining passes are 70 seconds, with the projected
    // five-percent decoder allowance retained until that phase begins.
    #expect(standard.remaining(elapsed: 20, projectedTotal: 120) == 76)

    var singlePhase = GenerationProgressTracker(profile: .singlePhase)
    #expect(singlePhase.remaining(elapsed: 1, projectedTotal: 20) == 19)
    singlePhase.record(phase: "speaking", phaseFraction: 0.25, elapsed: 4)
    #expect(singlePhase.remaining(elapsed: 4, projectedTotal: 20) == 12)
  }

  /// Sound effects and speech report one phase and no counter. Banding them
  /// would pin the bar at the preparation share for the entire run.
  @Test("A model that reports no pass counter uses its phase fraction")
  func singlePhaseModels() {
    var tracker = GenerationProgressTracker(countsPasses: false)
    tracker.record(phase: "denoise", phaseFraction: 0.5, elapsed: 10)
    #expect(tracker.overall == 0.5)
    tracker.record(phase: "denoise", phaseFraction: 1, elapsed: 20)
    #expect(tracker.overall == 1)

    var speech = GenerationProgressTracker(countsPasses: false)
    speech.record(phase: "speaking", phaseFraction: 0.25, elapsed: 4)
    #expect(speech.overall == 0.25)
    // It projects across the whole run, having no preparation to exclude.
    #expect(speech.remaining(elapsed: 4) == 12)
  }

  /// H3 subdivides a pass by transformer stage and LTX subdivides it by its 48
  /// blocks, so their fractions are genuinely within the named step.
  @Test("A subdivided pass advances within its own slice")
  func subdividedPasses() {
    var tracker = GenerationProgressTracker()
    tracker.record(
      phase: "denoise step 3/6 transformer", phaseFraction: 0, elapsed: 10)
    let atStart = tracker.overall
    tracker.record(
      phase: "denoise step 3/6 transformer", phaseFraction: 0.5, elapsed: 11)
    #expect(tracker.overall > atStart)
    tracker.record(
      phase: "denoise step 4/6 transformer", phaseFraction: 0, elapsed: 12)
    #expect(tracker.overall >= atStart)
  }

  /// H3 brackets each detailed transformer pass with a run-wide `denoise`
  /// event. Mistaking the first completed pass for the following decoder
  /// phase jumped the bar to 95% as step two began and pinned it there.
  @Test("H3 run-wide denoise ticks remain inside the denoising band")
  func h3RunWideDenoiseTicks() {
    var tracker = GenerationProgressTracker()

    tracker.record(phase: "denoise", phaseFraction: 0, elapsed: 1)
    tracker.record(
      phase: "denoise step 1/8 transformer", phaseFraction: 1, elapsed: 2)
    let afterFirstPass = tracker.overall
    tracker.record(phase: "denoise", phaseFraction: 1.0 / 8, elapsed: 2)
    tracker.record(
      phase: "denoise step 2/8 transformer", phaseFraction: 0, elapsed: 3)

    #expect(tracker.overall == afterFirstPass)
    #expect(tracker.overall < GenerationRemaining.decodeProgress)
    #expect(!tracker.isFinishing)

    tracker.record(phase: "denoise", phaseFraction: 1, elapsed: 10)
    #expect(tracker.overall == GenerationRemaining.decodeProgress)
    #expect(tracker.isFinishing)
  }

  /// The reported 15-second run spent 16.4% of its wall time in the video VAE.
  /// Its bar used to jump from the end of denoising to a fixed 95% and stay
  /// there for 13.5 minutes with no ETA.
  @Test("LTX video VAE advances through its own band with a phase ETA")
  func ltxVideoVAEProgress() {
    var tracker = GenerationProgressTracker(profile: .ltx)
    tracker.record(phase: "text encoder", phaseFraction: 1, elapsed: 17)
    #expect(abs(tracker.overall - 0.02) < 1e-12)

    tracker.record(phase: "denoise step 8/8", phaseFraction: 1, elapsed: 4_100)
    #expect(abs(tracker.overall - 0.83) < 1e-12)

    tracker.record(phase: "video VAE", phaseFraction: 0, elapsed: 4_100)
    #expect(abs(tracker.overall - 0.83) < 1e-12)
    #expect(tracker.remaining(elapsed: 4_100) == nil)
    #expect(!tracker.isFinishing)

    tracker.record(phase: "video VAE", phaseFraction: 0.5, elapsed: 4_500)
    #expect(abs(tracker.overall - 0.91) < 1e-12)
    #expect(tracker.remaining(elapsed: 4_500) == 400)
    #expect(!tracker.isFinishing)

    tracker.record(phase: "video VAE", phaseFraction: 1, elapsed: 4_900)
    #expect(abs(tracker.overall - 0.99) < 1e-12)
    #expect(tracker.remaining(elapsed: 4_900) == nil)
    #expect(tracker.isFinishing)

    tracker.record(phase: "vocoder", phaseFraction: 0, elapsed: 4_901)
    #expect(abs(tracker.overall - 0.99) < 1e-12)
    tracker.finish()
    #expect(tracker.overall == 1)
  }

  @Test("The standard profile keeps the existing image decoder band")
  func standardProfileCompatibility() {
    var tracker = GenerationProgressTracker(profile: .standard)
    tracker.record(phase: "denoise step 8/8", phaseFraction: 1, elapsed: 100)
    tracker.record(phase: "image VAE", phaseFraction: 0.5, elapsed: 101)
    #expect(tracker.overall == GenerationRemaining.decodeProgress)
    #expect(tracker.isFinishing)
  }
}
