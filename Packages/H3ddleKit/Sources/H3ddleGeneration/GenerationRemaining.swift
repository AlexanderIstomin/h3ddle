import Foundation

/// Estimating how much of a generation is left.
///
/// The engine reports progress per phase — "denoise step 3/6 transformer"
/// counts transformer blocks and restarts every step — so a naive
/// elapsed-over-fraction projection describes the end of the current step
/// rather than the run. Denoising steps dominate the total, so the run's
/// position is read from the step counter in the phase name, with the
/// within-phase fraction filling in the current step.
public enum GenerationRemaining {
  /// The share of the bar given to everything before the first denoising
  /// pass — loading weights and encoding the prompt.
  ///
  /// It is a fixed slice rather than a measured one because preparation is a
  /// roughly constant cost while denoising grows with canvas and step count,
  /// so its true share is different every run and cannot be known at the
  /// start. Fifteen percent is about what it is for a 1024-pixel Z-Image
  /// render, which is the case where the wait was long enough to complain
  /// about: two minutes of a bar that had not moved.
  public static let preparationShare = 0.15

  /// The share held back for decoding, which happens after the last pass.
  /// Without it the bar reaches 100% while the picture is still being made,
  /// and the last stretch of the wait happens against a full bar.
  public static let decodeShare = 0.05

  /// Where the bar sits once denoising is done and the decoder is running.
  public static let decodeProgress = 1 - decodeShare

  /// Overall completion in 0...1, or nil when the phase carries no step
  /// counter to anchor against.
  ///
  /// Denoising occupies the middle: above `preparationShare` so the bar never
  /// goes backwards when the first pass lands, and below `decodeProgress` so
  /// there is somewhere left to go afterwards.
  public static func overallProgress(
    phase: String,
    phaseFraction: Double
  ) -> Double? {
    guard let fraction = denoiseFraction(phase: phase, phaseFraction: phaseFraction)
    else { return nil }
    return overallProgress(denoiseFraction: fraction)
  }

  /// Maps a run-wide denoising fraction into the denoising band. H3 emits
  /// these between its more detailed per-block events as a bare `denoise`
  /// phase, while fully GPU-resident runs call the phase `denoise enqueue`.
  public static func overallProgress(denoiseFraction: Double) -> Double {
    let fraction = min(max(denoiseFraction, 0), 1)
    return preparationShare + (decodeProgress - preparationShare) * fraction
  }

  /// How far through the *denoising* the run is, which is what a projection
  /// may extrapolate from. Preparation is deliberately excluded: it runs at
  /// its own unrelated pace, and a projection made from it says four minutes
  /// where the answer is fifteen.
  public static func denoiseFraction(
    phase: String,
    phaseFraction: Double
  ) -> Double? {
    guard let (step, total) = denoiseStep(in: phase), total > 0 else {
      return nil
    }
    let completedSteps = Double(step - 1)
    let fraction = min(max(phaseFraction, 0), 1)
    return min(1, max(0, (completedSteps + fraction) / Double(total)))
  }

  /// Where the bar sits while the run is still preparing. Real work drives
  /// it — layers of the text encoder — rather than a timer, so a stall shows
  /// as a stall instead of a bar that keeps moving over a hung process.
  public static func preparationProgress(phaseFraction: Double) -> Double {
    preparationShare * min(1, max(0, phaseFraction))
  }

  /// Pulls "3" and "6" out of "denoise step 3/6 transformer".
  static func denoiseStep(in phase: String) -> (step: Int, total: Int)? {
    guard let range = phase.range(of: "denoise step ") else { return nil }
    let rest = phase[range.upperBound...]
    let parts = rest.split(separator: " ", maxSplits: 1).first?.split(separator: "/")
    guard let parts, parts.count == 2,
      let step = Int(parts[0]), let total = Int(parts[1])
    else { return nil }
    return (step, total)
  }

  /// Straight-line projection from the pace so far. Returns nil while the
  /// estimate would be too noisy to show.
  public static func estimate(
    elapsed: TimeInterval,
    progress: Double,
    minimumProgress: Double = 0.05
  ) -> TimeInterval? {
    guard progress >= minimumProgress, progress < 1, elapsed > 1 else {
      return nil
    }
    let remaining = elapsed / progress - elapsed
    return remaining.isFinite && remaining > 0 ? remaining : nil
  }

  /// Counts down a pre-run end-to-end projection while the current run has
  /// not advanced far enough to measure its own pace. Invalid or exhausted
  /// projections return nil instead of turning into a misleading countdown.
  public static func projected(
    total: TimeInterval?,
    elapsed: TimeInterval
  ) -> TimeInterval? {
    guard let total, total.isFinite, total > 0, elapsed.isFinite else {
      return nil
    }
    let remaining = total - max(0, elapsed)
    return remaining > 0 ? remaining : nil
  }

  /// Compact by necessity: this sits inside the progress ring, where a long
  /// phrase overlaps the circle. Coarse by choice: the pipeline is uneven,
  /// so "~7 min" is honest where "6:47" would not be.
  public static func phrase(_ remaining: TimeInterval) -> String {
    guard remaining.isFinite, remaining > 0 else { return "almost done" }
    if remaining < 45 {
      let seconds = max(5, Int((remaining / 5).rounded()) * 5)
      return "~\(seconds) s"
    }
    let minutes = Int((remaining / 60).rounded())
    if minutes < 60 { return "~\(max(1, minutes)) min" }
    let hours = remaining / 3600
    let rounded = (hours * 2).rounded() / 2
    return rounded == rounded.rounded()
      ? "~\(Int(rounded)) h"
      : String(format: "~%.1f h", rounded)
  }
}
