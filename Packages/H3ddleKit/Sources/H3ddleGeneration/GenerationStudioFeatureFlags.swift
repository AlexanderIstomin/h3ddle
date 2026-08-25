import Foundation

/// Experimental studio controls that should not become part of the default
/// generation contract merely because their engine support is still useful
/// for development and comparisons.
public struct GenerationStudioFeatureFlags: Equatable, Sendable {
  public static let advancedH3ControlsKey = "H3DDLE_ENABLE_H3_ADVANCED_CONTROLS"
  public static let h3MaskedSourceKey = "H3DDLE_ENABLE_H3_MASKED_SOURCE"

  public let advancedH3Controls: Bool
  public let h3MaskedSource: Bool

  public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
    advancedH3Controls = Self.isEnabled(environment[Self.advancedH3ControlsKey])
    h3MaskedSource = Self.isEnabled(environment[Self.h3MaskedSourceKey])
  }

  /// Hidden controls cannot leave a saved experimental value active. This is
  /// deliberately separate from persistence: setting the flag again restores
  /// the user's previous selection for another comparison.
  public func effectiveActiveDiTLayers(_ requested: Int) -> Int {
    advancedH3Controls ? requested : 50
  }

  public func effectiveCoreReuse(_ requested: Int) -> Int {
    advancedH3Controls ? requested : 1
  }

  private static func isEnabled(_ value: String?) -> Bool {
    guard let value else { return false }
    switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "1", "true", "yes", "on": return true
    default: return false
    }
  }
}
