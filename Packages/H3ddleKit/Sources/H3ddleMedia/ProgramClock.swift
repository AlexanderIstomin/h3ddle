import Foundation

public struct ProgramClock: Hashable, Sendable {
  public static let programFramesPerSecond = 24.0

  public var currentTime: TimeInterval
  public var isPlaying: Bool
  public var isLooping: Bool
  public var framesPerSecond: Double

  public init(
    currentTime: TimeInterval = 0,
    isPlaying: Bool = false,
    isLooping: Bool = false,
    framesPerSecond: Double = ProgramClock.programFramesPerSecond
  ) {
    self.currentTime = max(0, currentTime)
    self.isPlaying = isPlaying
    self.isLooping = isLooping
    self.framesPerSecond = max(1, framesPerSecond)
  }

  public mutating func setTime(_ time: TimeInterval, duration: TimeInterval) {
    currentTime = Self.clamp(time, duration: duration)
  }

  public mutating func step(frames: Int, duration: TimeInterval) {
    setTime(currentTime + Double(frames) / framesPerSecond, duration: duration)
  }

  public mutating func skipToStart() {
    currentTime = 0
  }

  public mutating func skipToEnd(duration: TimeInterval) {
    currentTime = max(0, duration)
  }

  /// Advances the clock. Returns `false` when playback should stop.
  @discardableResult
  public mutating func advance(by delta: TimeInterval, duration: TimeInterval) -> Bool {
    guard isPlaying else { return false }
    guard duration > 0 else {
      currentTime = 0
      isPlaying = false
      return false
    }

    var next = currentTime + max(0, delta)
    if next >= duration {
      if isLooping {
        next = next.truncatingRemainder(dividingBy: duration)
        currentTime = next
        return true
      }
      currentTime = duration
      isPlaying = false
      return false
    }

    currentTime = next
    return true
  }

  public static func clamp(_ time: TimeInterval, duration: TimeInterval) -> TimeInterval {
    guard duration > 0 else { return 0 }
    return min(max(0, time), duration)
  }

  public func formattedTimecode() -> String {
    Self.formatTimecode(currentTime, framesPerSecond: framesPerSecond)
  }

  public static func formatTimecode(
    _ time: TimeInterval,
    framesPerSecond: Double = ProgramClock.programFramesPerSecond
  ) -> String {
    let fps = max(1, Int(framesPerSecond.rounded()))
    let totalFrames = max(0, Int((max(0, time) * Double(fps)).rounded()))
    let frames = totalFrames % fps
    let totalSeconds = totalFrames / fps
    let seconds = totalSeconds % 60
    let minutes = (totalSeconds / 60) % 60
    let hours = totalSeconds / 3_600
    return String(format: "%02d:%02d:%02d:%02d", hours, minutes, seconds, frames)
  }

  public static func formatShort(_ time: TimeInterval) -> String {
    let totalSeconds = max(0, Int(time.rounded()))
    return String(format: "%d:%02d", totalSeconds / 60, totalSeconds % 60)
  }
}
