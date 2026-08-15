import Foundation

public enum VisualTransitionKind: String, Codable, Sendable, Hashable, CaseIterable, Identifiable {
  case dissolve
  case fade
  case wipe

  public var id: String { rawValue }

  public var label: String {
    switch self {
    case .dissolve: "Dissolve"
    case .fade: "Fade"
    case .wipe: "Wipe"
    }
  }

  public var symbol: String {
    switch self {
    case .dissolve: "circle.lefthalf.filled"
    case .fade: "moon.fill"
    case .wipe: "rectangle.righthalf.inset.filled.arrow.right"
    }
  }
}

public struct VisualTransition: Hashable, Codable, Sendable {
  public var kind: VisualTransitionKind
  public var duration: TimeInterval

  public init(kind: VisualTransitionKind, duration: TimeInterval) {
    self.kind = kind
    self.duration = max(0, duration)
  }
}

public enum VisualTransitionMath: Sendable {
  public static let defaultDuration: TimeInterval = 0.5

  /// Overlap mix: the incoming clip starts `duration` earlier so the two clips
  /// share a window. The outgoing tail and incoming head are blended there.
  public static func maximumDuration(
    outgoing: TimeInterval,
    incoming: TimeInterval
  ) -> TimeInterval {
    min(max(0, outgoing), max(0, incoming))
  }

  public static func resolvedDuration(
    _ duration: TimeInterval,
    outgoing: TimeInterval,
    incoming: TimeInterval
  ) -> TimeInterval {
    min(max(0, duration), maximumDuration(outgoing: outgoing, incoming: incoming))
  }

  public static func progress(
    at time: TimeInterval,
    cut: TimeInterval,
    duration: TimeInterval
  ) -> Double? {
    let duration = max(0, duration)
    guard duration > 0.000_1 else { return nil }
    guard time >= cut, time < cut + duration else { return nil }
    return min(1, max(0, (time - cut) / duration))
  }
}
