import Foundation

public enum TextAlignment: String, Codable, Sendable {
  /// Visual left in LTR. RTL remapping is a later follow-up.
  case leading
  case center
  /// Visual right in LTR.
  case trailing
}

public enum TextWrapMode: String, Codable, Sendable {
  case none
  case wrap
}

public struct TextColor: Hashable, Codable, Sendable {
  public var r: Double
  public var g: Double
  public var b: Double
  public var a: Double

  public static let clear = TextColor(r: 0, g: 0, b: 0, a: 0)
  public static let white = TextColor(r: 1, g: 1, b: 1, a: 1)
  public static let black = TextColor(r: 0, g: 0, b: 0, a: 1)

  public init(r: Double, g: Double, b: Double, a: Double) {
    self.r = min(max(r, 0), 1)
    self.g = min(max(g, 0), 1)
    self.b = min(max(b, 0), 1)
    self.a = min(max(a, 0), 1)
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.init(
      r: try container.decodeIfPresent(Double.self, forKey: .r) ?? 0,
      g: try container.decodeIfPresent(Double.self, forKey: .g) ?? 0,
      b: try container.decodeIfPresent(Double.self, forKey: .b) ?? 0,
      a: try container.decodeIfPresent(Double.self, forKey: .a) ?? 1
    )
  }

  enum CodingKeys: String, CodingKey {
    case r, g, b, a
  }
}

public struct TextStyle: Hashable, Codable, Sendable {
  public var fontFamily: String
  public var fontPostScriptName: String?
  public var fontWeight: Int
  public var italic: Bool
  /// Project-settings points at `transform.scale == 1`.
  public var fontSize: Double
  public var alignment: TextAlignment
  public var fill: TextColor
  public var wrap: TextWrapMode
  /// Settings pixels. `nil` hugs content.
  public var boxWidth: Double?
  public var lineHeight: Double
  public var letterSpacing: Double
  public var strokeWidth: Double
  public var strokeColor: TextColor
  public var shadowOffsetX: Double
  public var shadowOffsetY: Double
  public var shadowBlur: Double
  public var shadowColor: TextColor
  public var backgroundColor: TextColor
  public var backgroundPadding: Double
  public var backgroundCornerRadius: Double

  public static let `default` = TextStyle()

  public init(
    fontFamily: String = ".AppleSystemUIFont",
    fontPostScriptName: String? = nil,
    fontWeight: Int = 400,
    italic: Bool = false,
    fontSize: Double = 48,
    alignment: TextAlignment = .center,
    fill: TextColor = .white,
    wrap: TextWrapMode = .none,
    boxWidth: Double? = nil,
    lineHeight: Double = 1.2,
    letterSpacing: Double = 0,
    strokeWidth: Double = 0,
    strokeColor: TextColor = .black,
    shadowOffsetX: Double = 0,
    shadowOffsetY: Double = 0,
    shadowBlur: Double = 0,
    shadowColor: TextColor = TextColor(r: 0, g: 0, b: 0, a: 0.5),
    backgroundColor: TextColor = .clear,
    backgroundPadding: Double = 0,
    backgroundCornerRadius: Double = 0
  ) {
    self.fontFamily = fontFamily
    self.fontPostScriptName = fontPostScriptName
    self.fontWeight = min(max(fontWeight, 100), 900)
    self.italic = italic
    self.fontSize = max(fontSize, 1)
    self.alignment = alignment
    self.fill = fill
    self.wrap = wrap
    self.boxWidth = boxWidth.map { max(0, $0) }
    self.lineHeight = max(lineHeight, 0.5)
    self.letterSpacing = letterSpacing
    self.strokeWidth = max(strokeWidth, 0)
    self.strokeColor = strokeColor
    self.shadowOffsetX = shadowOffsetX
    self.shadowOffsetY = shadowOffsetY
    self.shadowBlur = max(shadowBlur, 0)
    self.shadowColor = shadowColor
    self.backgroundColor = backgroundColor
    self.backgroundPadding = max(backgroundPadding, 0)
    self.backgroundCornerRadius = max(backgroundCornerRadius, 0)
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.init(
      fontFamily: try container.decodeIfPresent(String.self, forKey: .fontFamily)
        ?? ".AppleSystemUIFont",
      fontPostScriptName: try container.decodeIfPresent(
        String.self,
        forKey: .fontPostScriptName
      ),
      fontWeight: try container.decodeIfPresent(Int.self, forKey: .fontWeight) ?? 400,
      italic: try container.decodeIfPresent(Bool.self, forKey: .italic) ?? false,
      fontSize: try container.decodeIfPresent(Double.self, forKey: .fontSize) ?? 48,
      alignment: try container.decodeIfPresent(TextAlignment.self, forKey: .alignment)
        ?? .center,
      fill: try container.decodeIfPresent(TextColor.self, forKey: .fill) ?? .white,
      wrap: try container.decodeIfPresent(TextWrapMode.self, forKey: .wrap) ?? .none,
      boxWidth: try container.decodeIfPresent(Double.self, forKey: .boxWidth),
      lineHeight: try container.decodeIfPresent(Double.self, forKey: .lineHeight) ?? 1.2,
      letterSpacing: try container.decodeIfPresent(Double.self, forKey: .letterSpacing)
        ?? 0,
      strokeWidth: try container.decodeIfPresent(Double.self, forKey: .strokeWidth) ?? 0,
      strokeColor: try container.decodeIfPresent(TextColor.self, forKey: .strokeColor)
        ?? .black,
      shadowOffsetX: try container.decodeIfPresent(Double.self, forKey: .shadowOffsetX)
        ?? 0,
      shadowOffsetY: try container.decodeIfPresent(Double.self, forKey: .shadowOffsetY)
        ?? 0,
      shadowBlur: try container.decodeIfPresent(Double.self, forKey: .shadowBlur) ?? 0,
      shadowColor: try container.decodeIfPresent(TextColor.self, forKey: .shadowColor)
        ?? TextColor(r: 0, g: 0, b: 0, a: 0.5),
      backgroundColor: try container.decodeIfPresent(
        TextColor.self,
        forKey: .backgroundColor
      ) ?? .clear,
      backgroundPadding: try container.decodeIfPresent(
        Double.self,
        forKey: .backgroundPadding
      ) ?? 0,
      backgroundCornerRadius: try container.decodeIfPresent(
        Double.self,
        forKey: .backgroundCornerRadius
      ) ?? 0
    )
  }

  enum CodingKeys: String, CodingKey {
    case fontFamily
    case fontPostScriptName
    case fontWeight
    case italic
    case fontSize
    case alignment
    case fill
    case wrap
    case boxWidth
    case lineHeight
    case letterSpacing
    case strokeWidth
    case strokeColor
    case shadowOffsetX
    case shadowOffsetY
    case shadowBlur
    case shadowColor
    case backgroundColor
    case backgroundPadding
    case backgroundCornerRadius
  }
}

public struct TextItem: Identifiable, Hashable, Codable, Sendable {
  public let id: UUID
  public var startTime: TimeInterval
  public var duration: TimeInterval
  public var isEnabled: Bool
  public var text: String
  public var style: TextStyle
  public var canvasTransform: CanvasObjectTransform

  public var endTime: TimeInterval { startTime + duration }

  public init(
    id: UUID = UUID(),
    startTime: TimeInterval,
    duration: TimeInterval,
    isEnabled: Bool = true,
    text: String = "Text",
    style: TextStyle = .default,
    canvasTransform: CanvasObjectTransform = .identity
  ) {
    self.id = id
    self.startTime = max(0, startTime)
    self.duration = max(0, duration)
    self.isEnabled = isEnabled
    self.text = text
    self.style = style
    self.canvasTransform = canvasTransform
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    id = try container.decode(UUID.self, forKey: .id)
    startTime = max(0, try container.decode(TimeInterval.self, forKey: .startTime))
    duration = max(0, try container.decode(TimeInterval.self, forKey: .duration))
    isEnabled = try container.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? true
    text = try container.decodeIfPresent(String.self, forKey: .text) ?? "Text"
    style = try container.decodeIfPresent(TextStyle.self, forKey: .style) ?? .default
    canvasTransform =
      try container.decodeIfPresent(CanvasObjectTransform.self, forKey: .canvasTransform)
      ?? .identity
  }

  enum CodingKeys: String, CodingKey {
    case id
    case startTime
    case duration
    case isEnabled
    case text
    case style
    case canvasTransform
  }
}
