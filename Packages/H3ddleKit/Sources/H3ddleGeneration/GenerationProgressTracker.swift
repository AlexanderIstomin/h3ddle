import Foundation

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

  /// Whether this run's engine counts denoising passes. Speech and sound
  /// effects run in one phase and report no counter; for them the phase
  /// fraction is the run's own, and banding it would pin the bar at 15%.
  public let countsPasses: Bool

  public init(countsPasses: Bool = true) {
    self.countsPasses = countsPasses
  }

  public mutating func record(
    phase: String,
    phaseFraction: Double,
    elapsed: TimeInterval
  ) {
    self.phase = phase
    let reached: Double
    if let banded = GenerationRemaining.overallProgress(
      phase: phase, phaseFraction: phaseFraction)
    {
      denoise = GenerationRemaining.denoiseFraction(
        phase: phase, phaseFraction: phaseFraction)
      reached = banded
    } else if !countsPasses {
      reached = min(1, max(0, phaseFraction))
    } else if denoise == nil {
      // Still preparing, and still measuring how long that is taking.
      preparationElapsed = elapsed
      reached = GenerationRemaining.preparationProgress(phaseFraction: phaseFraction)
    } else {
      // Past the last pass: the decoder is finishing the picture.
      reached = GenerationRemaining.decodeProgress
    }
    overall = max(overall, reached)
  }

  /// Seconds left, or nil while a projection would be too noisy to show.
  public func remaining(elapsed: TimeInterval) -> TimeInterval? {
    // Both halves have to measure the same stretch of the run.
    let basis: (elapsed: TimeInterval, progress: Double)? =
      if let denoise {
        (elapsed - preparationElapsed, denoise)
      } else if !countsPasses {
        (elapsed, overall)
      } else {
        nil
      }
    guard let basis else { return nil }
    return GenerationRemaining.estimate(
      elapsed: basis.elapsed, progress: basis.progress)
  }

  /// Whether the passes are done and only the decoder is left.
  public var isFinishing: Bool {
    overall >= GenerationRemaining.decodeProgress
  }

  /// The run produced its asset. The decode band has no inner counter to
  /// advance through, so the bar would otherwise stop at 95% and be replaced
  /// by the result rather than completing.
  public mutating func finish() {
    overall = 1
  }
}
