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
  /// Overall completion in 0...1, or nil when the phase carries no step
  /// counter to anchor against.
  public static func overallProgress(
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

  /// Compact by necessity: this sits inside the progress ring, where a long
  /// phrase overlaps the circle. Coarse by choice: the pipeline is uneven,
  /// so "~7 min" is honest where "6:47" would not be.
  public static func phrase(_ remaining: TimeInterval) -> String {
    guard remaining.isFinite, remaining > 0 else { return "almost done" }
    if remaining < 45 { return "~1 min" }
    let minutes = Int((remaining / 60).rounded())
    if minutes < 60 { return "~\(max(1, minutes)) min" }
    let hours = remaining / 3600
    let rounded = (hours * 2).rounded() / 2
    return rounded == rounded.rounded()
      ? "~\(Int(rounded)) h"
      : String(format: "~%.1f h", rounded)
  }
}
