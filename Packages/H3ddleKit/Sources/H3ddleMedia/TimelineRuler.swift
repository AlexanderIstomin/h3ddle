import Foundation

public struct TimelineTick: Equatable, Sendable {
  public var time: TimeInterval
  public var isMajor: Bool
  public var label: String

  public init(time: TimeInterval, isMajor: Bool, label: String) {
    self.time = time
    self.isMajor = isMajor
    self.label = label
  }
}

public enum TimelineRuler {
  public static let basePointsPerSecond: Double = 13
  public static let minimumZoom = 0.05
  public static let maximumZoom = 8.0
  public static let sliderMinimumZoom = 0.4
  public static let sliderMaximumZoom = 3.0
  public static let fallbackWindow: TimeInterval = 180
  /// Extra lane width so the last `m:ss` label is not clipped.
  public static let trailingLabelPadding: Double = 44

  public static func clampZoom(_ zoom: Double) -> Double {
    min(max(zoom, minimumZoom), maximumZoom)
  }

  public static func pointsPerSecond(zoom: Double) -> Double {
    basePointsPerSecond * clampZoom(zoom)
  }

  /// Keeps the timeline time under `anchor` (points from the lane's leading edge)
  /// stationary when zoom changes.
  public static func anchoredScrollOffset(
    currentZoom: Double,
    nextZoom: Double,
    scrollOffset: Double,
    anchor: Double
  ) -> Double {
    let oldPps = pointsPerSecond(zoom: currentZoom)
    let newPps = pointsPerSecond(zoom: nextZoom)
    let time = (scrollOffset + anchor) / oldPps
    return max(0, time * newPps - anchor)
  }

  public static func contentDuration(
    visualDuration: TimeInterval,
    audioTrackEnd: TimeInterval,
    textTrackEnd: TimeInterval = 0
  ) -> TimeInterval {
    max(visualDuration, audioTrackEnd, textTrackEnd, fallbackWindow)
  }

  public static func majorInterval(pointsPerSecond: Double) -> TimeInterval {
    let pps = max(1, pointsPerSecond)
    let steps = [1.0, 2, 5, 10, 15, 30, 60, 120]
    return steps.first { $0 * pps > 60 } ?? 120
  }

  public static func minorInterval(pointsPerSecond: Double) -> TimeInterval {
    majorInterval(pointsPerSecond: pointsPerSecond) / 4
  }

  public static func drawsMinorTicks(pointsPerSecond: Double) -> Bool {
    minorInterval(pointsPerSecond: pointsPerSecond) * max(1, pointsPerSecond) >= 7
  }

  public static func ticks(duration: TimeInterval, pointsPerSecond: Double) -> [TimelineTick] {
    let pps = max(1, pointsPerSecond)
    let major = majorInterval(pointsPerSecond: pps)
    let minor = major / 4
    let drawMinor = drawsMinorTicks(pointsPerSecond: pps)
    let end = max(0, duration)
    var ticks: [TimelineTick] = []
    var index = 0
    while Double(index) * minor <= end + 1e-6 {
      let time = Double(index) * minor
      let isMajor = abs((time / major) - (time / major).rounded()) < 1e-6
      if isMajor || drawMinor {
        ticks.append(
          TimelineTick(
            time: time,
            isMajor: isMajor,
            label: isMajor ? ProgramClock.formatShort(time) : ""
          )
        )
      }
      index += 1
    }
    return ticks
  }
}
