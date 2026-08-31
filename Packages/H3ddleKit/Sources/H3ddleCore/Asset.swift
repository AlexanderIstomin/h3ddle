import Foundation

public struct AssetID: Hashable, Codable, Sendable, RawRepresentable {
  public let rawValue: UUID

  public init(rawValue: UUID) {
    self.rawValue = rawValue
  }

  public init() {
    self.init(rawValue: UUID())
  }

  public init?(uuidString: String) {
    guard let id = UUID(uuidString: uuidString) else { return nil }
    self.init(rawValue: id)
  }
}

public enum MediaKind: String, Codable, CaseIterable, Sendable {
  case video
  case image
  case audio

  public var isVisual: Bool {
    self == .video || self == .image
  }
}

public enum AssetMetadataKey {
  /// A versioned, path-free recipe describing how a generated asset was made.
  /// The value is deliberately untyped here so H3ddleCore stays independent
  /// from the generation module while project interchange can round-trip it.
  public static let generationRecipe = "h3ddleGeneration"
}

public struct AssetReference: Identifiable, Hashable, Codable, Sendable {
  public let id: AssetID
  public var kind: MediaKind
  public var displayName: String
  public var url: URL
  public var duration: TimeInterval
  public var metadata: [String: JSONValue]

  public init(
    id: AssetID = AssetID(),
    kind: MediaKind,
    displayName: String,
    url: URL,
    duration: TimeInterval,
    metadata: [String: JSONValue] = [:]
  ) {
    self.id = id
    self.kind = kind
    self.displayName = displayName
    self.url = url
    self.duration = max(0, duration)
    self.metadata = metadata
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    id = try container.decode(AssetID.self, forKey: .id)
    kind = try container.decode(MediaKind.self, forKey: .kind)
    displayName = try container.decode(String.self, forKey: .displayName)
    url = try container.decode(URL.self, forKey: .url)
    duration = max(0, try container.decode(TimeInterval.self, forKey: .duration))
    metadata = try container.decodeIfPresent(
      [String: JSONValue].self,
      forKey: .metadata
    ) ?? [:]
  }

  enum CodingKeys: String, CodingKey {
    case id
    case kind
    case displayName
    case url
    case duration
    case metadata
  }
}
