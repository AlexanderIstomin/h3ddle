import Foundation

public enum VisualEffectKind: String, Codable, Sendable, Hashable, CaseIterable, Identifiable {
  case colorGrade
  case vignette
  case filmGrain
  case sharpen
  case blur
  case bloom
  case chromaKey

  public var id: String { rawValue }

  public var label: String {
    switch self {
    case .colorGrade: "Color grade"
    case .vignette: "Vignette"
    case .filmGrain: "Film grain"
    case .sharpen: "Sharpen"
    case .blur: "Blur"
    case .bloom: "Bloom"
    case .chromaKey: "Chroma key"
    }
  }

  public var category: VisualEffectCategory {
    switch self {
    case .colorGrade, .vignette: .color
    case .filmGrain, .sharpen, .blur, .bloom: .stylize
    case .chromaKey: .key
    }
  }
}

public enum VisualEffectCategory: String, Codable, Sendable, Hashable, CaseIterable, Identifiable {
  case color
  case stylize
  case key

  public var id: String { rawValue }

  public var label: String {
    switch self {
    case .color: "Color"
    case .stylize: "Stylize"
    case .key: "Key"
    }
  }
}

public struct VisualEffectInstance: Identifiable, Hashable, Codable, Sendable {
  public let id: UUID
  public var kind: VisualEffectKind
  public var isEnabled: Bool
  public var parameters: [String: Double]

  public init(
    id: UUID = UUID(),
    kind: VisualEffectKind,
    isEnabled: Bool = true,
    parameters: [String: Double]? = nil
  ) {
    self.id = id
    self.kind = kind
    self.isEnabled = isEnabled
    self.parameters = parameters ?? VisualEffectCatalog.defaultParameters(for: kind)
  }

  public func copying() -> VisualEffectInstance {
    VisualEffectInstance(kind: kind, isEnabled: isEnabled, parameters: parameters)
  }

  public func value(_ key: String, default defaultValue: Double = 0) -> Double {
    parameters[key] ?? defaultValue
  }
}

public enum VisualEffectCatalog: Sendable {
  public static func defaultParameters(for kind: VisualEffectKind) -> [String: Double] {
    switch kind {
    case .colorGrade:
      ["exposure": 0, "contrast": 1, "saturation": 1]
    case .vignette:
      ["amount": 0.5, "feather": 0.5]
    case .filmGrain:
      ["amount": 0.35, "size": 0.5]
    case .sharpen:
      ["amount": 0.4]
    case .blur:
      ["radius": 4]
    case .bloom:
      ["amount": 0.5, "threshold": 0.5]
    case .chromaKey:
      ["hue": 0, "softness": 0.25]
    }
  }

  public static func knobs(for kind: VisualEffectKind) -> [VisualEffectKnob] {
    switch kind {
    case .colorGrade:
      [
        VisualEffectKnob(key: "exposure", label: "Exposure", range: -2...2),
        VisualEffectKnob(key: "contrast", label: "Contrast", range: 0.4...1.8),
        VisualEffectKnob(key: "saturation", label: "Saturation", range: 0...2),
      ]
    case .vignette:
      [
        VisualEffectKnob(key: "amount", label: "Amount", range: 0...1),
        VisualEffectKnob(key: "feather", label: "Feather", range: 0...1),
      ]
    case .filmGrain:
      [
        VisualEffectKnob(key: "amount", label: "Amount", range: 0...1),
        VisualEffectKnob(key: "size", label: "Size", range: 0...1),
      ]
    case .sharpen:
      [VisualEffectKnob(key: "amount", label: "Amount", range: 0...1)]
    case .blur:
      [VisualEffectKnob(key: "radius", label: "Radius", range: 0...24)]
    case .bloom:
      [
        VisualEffectKnob(key: "amount", label: "Amount", range: 0...1),
        VisualEffectKnob(key: "threshold", label: "Threshold", range: 0...1),
      ]
    case .chromaKey:
      [
        VisualEffectKnob(key: "hue", label: "Key", range: 0...1),
        VisualEffectKnob(key: "softness", label: "Softness", range: 0...1),
      ]
    }
  }
}

public struct VisualEffectKnob: Hashable, Sendable, Identifiable {
  public var key: String
  public var label: String
  public var range: ClosedRange<Double>

  public var id: String { key }

  public init(key: String, label: String, range: ClosedRange<Double>) {
    self.key = key
    self.label = label
    self.range = range
  }
}
