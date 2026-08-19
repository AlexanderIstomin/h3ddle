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

public enum TimelineFilmstripSampling {
  public static let maximumFrameCount = 12

  public static func times(
    sourceOffset: TimeInterval,
    duration: TimeInterval,
    frameCount: Int = maximumFrameCount
  ) -> [TimeInterval] {
    let start = max(0, sourceOffset)
    let count = min(max(1, frameCount), maximumFrameCount)
    let sampledDuration = max(0, duration - 1.0 / 600.0)
    guard count > 1, sampledDuration > 0 else {
      return [start + sampledDuration / 2]
    }
    return (0..<count).map { index in
      start + sampledDuration * Double(index) / Double(count - 1)
    }
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

  public static func visibleTimeRange(
    scrollOffset: Double,
    viewportWidth: Double,
    pointsPerSecond: Double,
    overscanPoints: Double = 0
  ) -> ClosedRange<TimeInterval> {
    let pps = max(pointsPerSecond, 0.000_1)
    let overscan = max(0, overscanPoints)
    let startPoints = max(0, scrollOffset - overscan)
    let endPoints = max(
      startPoints,
      scrollOffset + max(0, viewportWidth) + overscan
    )
    return startPoints / pps...endPoints / pps
  }

  public static func spanIntersects(
    start: TimeInterval,
    duration: TimeInterval,
    visibleRange: ClosedRange<TimeInterval>
  ) -> Bool {
    start <= visibleRange.upperBound
      && start + max(0, duration) >= visibleRange.lowerBound
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
    ticks(
      duration: duration,
      pointsPerSecond: pointsPerSecond,
      visibleRange: 0...max(0, duration)
    )
  }

  public static func ticks(
    duration: TimeInterval,
    pointsPerSecond: Double,
    visibleRange: ClosedRange<TimeInterval>
  ) -> [TimelineTick] {
    let pps = max(1, pointsPerSecond)
    let major = majorInterval(pointsPerSecond: pps)
    let minor = major / 4
    let drawMinor = drawsMinorTicks(pointsPerSecond: pps)
    let end = max(0, duration)
    let visibleStart = max(0, visibleRange.lowerBound)
    let visibleEnd = min(end, visibleRange.upperBound)
    guard visibleStart <= visibleEnd else { return [] }
    let firstIndex = max(0, Int(ceil((visibleStart - 1e-6) / minor)))
    let lastIndex = Int(floor((visibleEnd + 1e-6) / minor))
    guard firstIndex <= lastIndex else { return [] }
    var ticks: [TimelineTick] = []
    for index in firstIndex...lastIndex {
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
    }
    return ticks
  }
}
