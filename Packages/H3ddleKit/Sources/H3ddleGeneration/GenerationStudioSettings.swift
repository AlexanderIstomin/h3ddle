import Foundation
import H3ddleEngineProtocol

/// Output size, named by the short edge the way video resolutions normally
/// are: a tier means the same vertical detail whatever the aspect, where
/// "512x512" describes only the square case and misleads for 9:16.
///
/// The ladder starts at 352p because that is the smallest canvas the
/// reference workflows document (0.2 megapixels at 16:9) — below it H3 stops
/// following prompts reliably, measured repeatedly at 256 square. Long edges
/// are snapped to the multiples of 32 the engine requires, which reproduces
/// the published resolution table from 480p up: 864x480, 1024x576, 1376x768,
/// 1920x1088. Only 352p differs, because that table targets megapixels rather
/// than a fixed short edge.
public enum GenerationCanvas: String, CaseIterable, Codable, Sendable, Identifiable {
  case p352
  case p480
  case p576
  case p768
  case p1088

  public var id: String { rawValue }

  /// The fixed dimension: height in landscape, width in portrait.
  public var shortEdge: Int {
    switch self {
    case .p352: 352
    case .p480: 480
    case .p576: 576
    case .p768: 768
    case .p1088: 1088
    }
  }

  public var label: String { "\(shortEdge)p" }

  public var engineQuality: EngineGenerationQuality {
    switch self {
    case .p352, .p480: .preview
    case .p576: .standard
    case .p768, .p1088: .high
    }
  }

  /// Rounds to the nearest legal canvas step, never below one step.
  private static func snapped(_ value: Double) -> Int {
    max(32, Int((value / 32).rounded()) * 32)
  }

  /// `aspect` is width divided by height. The short edge is exact; the long
  /// edge snaps, so extreme ratios drift slightly rather than being refused.
  public func dimensions(aspect: Double) -> (width: Int, height: Int) {
    guard aspect.isFinite, aspect > 0 else {
      return (shortEdge, shortEdge)
    }
    if aspect >= 1 {
      return (Self.snapped(Double(shortEdge) * aspect), shortEdge)
    }
    return (shortEdge, Self.snapped(Double(shortEdge) / aspect))
  }

  /// Megapixels at this aspect, for showing the cost of a tier.
  public func megapixels(aspect: Double) -> Double {
    let size = dimensions(aspect: aspect)
    return Double(size.width * size.height) / 1_000_000
  }

  public init(from decoder: Decoder) throws {
    let raw = try decoder.singleValueContainer().decode(String.self)
    // Settings saved before the ladder was named by short edge.
    switch raw {
    case "square256", "square512": self = .p480
    case "native768": self = .p768
    case "native1344": self = .p1088
    default:
      guard let value = GenerationCanvas(rawValue: raw) else {
        throw DecodingError.dataCorruptedError(
          in: try decoder.singleValueContainer(),
          debugDescription: "unknown canvas \(raw)"
        )
      }
      self = value
    }
  }
}

public enum GenerationPreset: String, CaseIterable, Codable, Sendable, Identifiable {
  case preview
  case standard
  case high
  case custom

  public var id: String { rawValue }

  public var label: String {
    switch self {
    case .preview: "Preview"
    case .standard: "Standard"
    case .high: "High"
    case .custom: "Custom"
    }
  }

  public static let named: [GenerationPreset] = [.preview, .standard, .high]
}

public struct GenerationKnobSnapshot: Hashable, Codable, Sendable {
  public var canvas: GenerationCanvas
  public var denoisingSteps: Int
  public var activeDiTLayers: Int
  public var coreReuse: Int
  /// Replay cached tail-block residuals on schedule-gated steps. Faster at
  /// standard step counts; the result is a different sample of the same
  /// quality, so it stays a deliberate choice.
  public var blockCache: Bool
  /// Render stills from a 5-frame clip rather than 22. Roughly 3x faster and
  /// visibly softer, so the detailed path stays the default.
  public var fastStill: Bool

  public init(
    canvas: GenerationCanvas,
    denoisingSteps: Int,
    activeDiTLayers: Int,
    coreReuse: Int,
    blockCache: Bool = false,
    fastStill: Bool = false
  ) {
    self.canvas = canvas
    self.denoisingSteps = denoisingSteps
    self.activeDiTLayers = activeDiTLayers
    self.coreReuse = coreReuse
    self.blockCache = blockCache
    self.fastStill = fastStill
  }

  enum CodingKeys: String, CodingKey {
    case canvas
    case denoisingSteps
    case activeDiTLayers
    case coreReuse
    case blockCache
    case fastStill
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    canvas = try container.decode(GenerationCanvas.self, forKey: .canvas)
    denoisingSteps = try container.decode(Int.self, forKey: .denoisingSteps)
    activeDiTLayers = try container.decode(Int.self, forKey: .activeDiTLayers)
    coreReuse = try container.decode(Int.self, forKey: .coreReuse)
    // Settings persisted before the knob existed decode to off.
    blockCache = try container.decodeIfPresent(Bool.self, forKey: .blockCache) ?? false
    fastStill = try container.decodeIfPresent(Bool.self, forKey: .fastStill) ?? false
  }

  public static func preset(_ preset: GenerationPreset) -> GenerationKnobSnapshot {
    switch preset {
    case .preview:
      GenerationKnobSnapshot(
        canvas: .p352, denoisingSteps: 4, activeDiTLayers: 50, coreReuse: 1
      )
    case .standard:
      GenerationKnobSnapshot(
        canvas: .p480, denoisingSteps: 20, activeDiTLayers: 45, coreReuse: 1
      )
    case .high:
      GenerationKnobSnapshot(
        canvas: .p768, denoisingSteps: 20, activeDiTLayers: 50, coreReuse: 1
      )
    case .custom:
      .preset(.preview)
    }
  }
}

public struct GenerationStudioSettings: Hashable, Codable, Sendable {
  public var preset: GenerationPreset
  public var knobs: GenerationKnobSnapshot
  public var custom: GenerationKnobSnapshot
  public var seed: UInt64
  public var duration: Double
  public var alignedDurationStep: Double

  public static func makeDefault(
    seed: UInt64 = UInt64.random(in: 1..<100_000_000)
  ) -> GenerationStudioSettings {
    let preview = GenerationKnobSnapshot.preset(.preview)
    return GenerationStudioSettings(
      preset: .preview,
      knobs: preview,
      custom: preview,
      seed: seed,
      duration: 3,
      alignedDurationStep: 0
    )
  }

  public init(
    preset: GenerationPreset,
    knobs: GenerationKnobSnapshot,
    custom: GenerationKnobSnapshot,
    seed: UInt64,
    duration: Double,
    alignedDurationStep: Double
  ) {
    self.preset = preset
    self.knobs = knobs
    self.custom = custom
    self.seed = seed
    self.duration = duration
    self.alignedDurationStep = alignedDurationStep
  }

  public mutating func apply(preset newPreset: GenerationPreset) {
    if newPreset == .custom {
      knobs = custom
      preset = .custom
      return
    }
    preset = newPreset
    knobs = .preset(newPreset)
  }

  public mutating func updateKnobs(_ mutate: (inout GenerationKnobSnapshot) -> Void) {
    mutate(&knobs)
    if let match = GenerationPreset.named.first(where: { GenerationKnobSnapshot.preset($0) == knobs })
    {
      preset = match
    } else {
      custom = knobs
      preset = .custom
    }
  }
}
