import Foundation

/// Accumulates per-phase wall-clock durations from generation progress
/// events. H3 runs phases sequentially, so a phase change closes the
/// previous phase; a phase name may recur and is recorded as a new entry.
public struct GenerationPhaseTimeline: Hashable, Sendable {
  public struct Entry: Hashable, Sendable {
    public var phase: String
    public var duration: TimeInterval

    public init(phase: String, duration: TimeInterval) {
      self.phase = phase
      self.duration = max(0, duration)
    }
  }

  public private(set) var entries: [Entry] = []
  private var currentPhase: String?
  private var currentPhaseStart: TimeInterval = 0

  public init() {}

  /// Records a progress event observed `elapsed` seconds into the run.
  public mutating func record(phase: String, elapsed: TimeInterval) {
    guard phase != currentPhase else { return }
    closeCurrentPhase(at: elapsed)
    currentPhase = phase
    currentPhaseStart = elapsed
  }

  /// Closes the open phase, if any, at the final elapsed time.
  public mutating func finish(elapsed: TimeInterval) {
    closeCurrentPhase(at: elapsed)
    currentPhase = nil
  }

  /// One line suitable for logging, e.g.
  /// `text encoder 12.3s · denoise 601.0s · video VAE 44.2s`.
  public var summary: String? {
    guard !entries.isEmpty else { return nil }
    return
      entries
      .map { String(format: "%@ %.1fs", $0.phase, $0.duration) }
      .joined(separator: " · ")
  }

  private mutating func closeCurrentPhase(at elapsed: TimeInterval) {
    guard let phase = currentPhase else { return }
    entries.append(Entry(phase: phase, duration: elapsed - currentPhaseStart))
  }
}
