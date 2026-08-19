import Foundation

/// Different engines spend materially different shares of a run outside the
/// sampler. LTX's convolutional video VAE was 16.4% of the recorded 15-second
/// run; treating it like the short image decoder makes 95% arrive far too soon.
public enum GenerationProgressProfile: Equatable, Sendable {
  case standard
  case ltx
  case singlePhase
}

/// Turns the engine's per-phase progress events into the single advancing
/// number a progress ring needs.
///
/// The engine reports where it is *within a phase* — "denoise step 3/8" at
/// fraction 1, then "image VAE" at fraction 0 — because that is what each
/// stage can honestly say about itself. Shown directly, that reads as a bar
/// which fills up on every pass and drops back to nothing at the next stage.
/// This maps those events onto the run:
///
///     0 ──── preparation ────┤ denoising ├──── decode ──── 1
///          loading, encoding    the passes    the picture
///
/// Two properties are deliberate. It only ever moves **forward**, so no
/// stage boundary can send the bar backwards. And the estimate is projected
/// from the passes alone, timed from where preparation ended — charging two
/// minutes of loading against one eighth of the denoising is what reported
/// fifteen minutes for a picture that took seventy-eight seconds.
public struct GenerationProgressTracker: Equatable, Sendable {
  /// The run's position in 0...1. This is what to display.
  public private(set) var overall: Double = 0
  /// How far through the denoising passes, or nil before the first one.
  public private(set) var denoise: Double?
  /// Elapsed when preparation ended, which is where pass timing starts.
  public private(set) var preparationElapsed: TimeInterval = 0
  /// The most recent phase name the engine reported.
  public private(set) var phase = ""
  /// How far through LTX's video VAE, when that phase reports real work.
  public private(set) var videoVAE: Double?
  /// Elapsed time at the first video-VAE event, for a phase-local ETA.
  public private(set) var videoVAEStartedElapsed: TimeInterval?

  /// Whether this run's engine counts denoising passes. Speech and sound
  /// effects run in one phase and report no counter; for them the phase
  /// fraction is the run's own, and banding it would pin the bar at 15%.
  public let countsPasses: Bool
  public let profile: GenerationProgressProfile

  public init(countsPasses: Bool = true) {
    self.countsPasses = countsPasses
    profile = countsPasses ? .standard : .singlePhase
  }

  public init(profile: GenerationProgressProfile) {
    self.profile = profile
    countsPasses = profile != .singlePhase
  }

  public mutating func record(
    phase: String,
    phaseFraction: Double,
    elapsed: TimeInterval
  ) {
    self.phase = phase
    let reached: Double
    if profile == .ltx, phase == "video VAE" {
      if videoVAEStartedElapsed == nil {
        videoVAEStartedElapsed = elapsed
      }
      let fraction = min(1, max(0, phaseFraction))
      videoVAE = fraction
      reached = denoiseEnd + (videoVAEEnd - denoiseEnd) * fraction
    } else if let fraction = GenerationRemaining.denoiseFraction(
      phase: phase, phaseFraction: phaseFraction)
    {
      denoise = fraction
      reached = overallProgress(denoiseFraction: fraction)
    } else if !countsPasses {
      reached = min(1, max(0, phaseFraction))
    } else if phase == "denoise" || phase == "denoise enqueue" {
      // H3 reports the whole sampler's completed/total count between the
      // detailed "denoise step N/M transformer" events. It is still inside
      // denoising, not the decoder phase that follows it.
      let fraction = min(1, max(0, phaseFraction))
      denoise = fraction
      reached = overallProgress(denoiseFraction: fraction)
    } else if denoise == nil {
      // Still preparing, and still measuring how long that is taking.
      preparationElapsed = elapsed
      reached = preparationShare * min(1, max(0, phaseFraction))
    } else {
      // LTX reports its long video decode above. Its vocoder and mux are short
      // but still keep the last one percent for actual completion.
      reached = profile == .ltx && phase == "vocoder" ? videoVAEEnd : denoiseEnd
    }
    overall = max(overall, reached)
  }

  /// Seconds left. A pre-run end-to-end projection supplies the initial
  /// countdown; as soon as this run has enough phase progress, its measured
  /// pace takes over. During denoising, the pre-run estimate still supplies
  /// the later decoder share because the sampler cannot measure work it has
  /// not reached yet.
  public func remaining(
    elapsed: TimeInterval,
    projectedTotal: TimeInterval? = nil
  ) -> TimeInterval? {
    let projected = GenerationRemaining.projected(
      total: projectedTotal, elapsed: elapsed)
    if profile == .ltx, phase == "video VAE",
      let videoVAE, let started = videoVAEStartedElapsed
    {
      return GenerationRemaining.estimate(
        elapsed: elapsed - started, progress: videoVAE
      ) ?? (isFinishing ? nil : projected)
    }
    // Both halves have to measure the same stretch of the run.
    let basis: (elapsed: TimeInterval, progress: Double)? =
      if let denoise {
        (elapsed - preparationElapsed, denoise)
      } else if !countsPasses {
        (elapsed, overall)
      } else {
        nil
      }
    guard let basis else { return isFinishing ? nil : projected }
    guard let measured = GenerationRemaining.estimate(
      elapsed: basis.elapsed, progress: basis.progress)
    else { return isFinishing ? nil : projected }

    if denoise != nil,
      let projectedTotal,
      projectedTotal.isFinite,
      projectedTotal > 0
    {
      return measured + projectedTotal * (1 - denoiseEnd)
    }
    return measured
  }

  /// Whether the passes are done and only the decoder is left.
  public var isFinishing: Bool {
    if profile == .ltx, phase == "video VAE", (videoVAE ?? 0) < 1 {
      return false
    }
    return overall >= denoiseEnd
  }

  /// The run produced its asset. Decoder bands deliberately retain at least
  /// one percent for the final write, so completion is the only event that may
  /// fill the bar completely.
  public mutating func finish() {
    overall = 1
  }

  private var preparationShare: Double {
    profile == .ltx ? 0.02 : GenerationRemaining.preparationShare
  }

  private var denoiseEnd: Double {
    profile == .ltx ? 0.83 : GenerationRemaining.decodeProgress
  }

  private var videoVAEEnd: Double { 0.99 }

  private func overallProgress(denoiseFraction: Double) -> Double {
    let fraction = min(1, max(0, denoiseFraction))
    return preparationShare + (denoiseEnd - preparationShare) * fraction
  }
}
