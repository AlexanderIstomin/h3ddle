import Foundation
import H3ddleEngineProtocol

public enum GenerationCanvas: String, CaseIterable, Codable, Sendable, Identifiable {
  case square256
  case square512
  case native768
  case native1344

  public var id: String { rawValue }

  public var engineQuality: EngineGenerationQuality {
    switch self {
    case .square256: .preview
    case .square512: .standard
    case .native768, .native1344: .high
    }
  }

  public func dimensions(isPortrait: Bool) -> (width: Int, height: Int) {
    switch self {
    case .square256: (256, 256)
    case .square512: (512, 512)
    case .native768: (768, 768)
    case .native1344: isPortrait ? (768, 1344) : (1344, 768)
    }
  }

  public func label(isPortrait: Bool) -> String {
    let size = dimensions(isPortrait: isPortrait)
    let pixels = "\(size.width)×\(size.height)"
    switch self {
    case .square256, .square512:
      return pixels
    case .native768, .native1344:
      return "\(pixels) · Native"
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

  public init(
    canvas: GenerationCanvas,
    denoisingSteps: Int,
    activeDiTLayers: Int,
    coreReuse: Int
  ) {
    self.canvas = canvas
    self.denoisingSteps = denoisingSteps
    self.activeDiTLayers = activeDiTLayers
    self.coreReuse = coreReuse
  }

  public static func preset(_ preset: GenerationPreset) -> GenerationKnobSnapshot {
    switch preset {
    case .preview:
      GenerationKnobSnapshot(
        canvas: .square256, denoisingSteps: 4, activeDiTLayers: 50, coreReuse: 1
      )
    case .standard:
      GenerationKnobSnapshot(
        canvas: .square512, denoisingSteps: 20, activeDiTLayers: 45, coreReuse: 1
      )
    case .high:
      GenerationKnobSnapshot(
        canvas: .native768, denoisingSteps: 20, activeDiTLayers: 50, coreReuse: 1
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
