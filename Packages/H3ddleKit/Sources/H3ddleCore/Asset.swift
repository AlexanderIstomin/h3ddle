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

public struct AssetReference: Identifiable, Hashable, Codable, Sendable {
  public let id: AssetID
  public var kind: MediaKind
  public var displayName: String
  public var url: URL
  public var duration: TimeInterval

  public init(
    id: AssetID = AssetID(),
    kind: MediaKind,
    displayName: String,
    url: URL,
    duration: TimeInterval
  ) {
    self.id = id
    self.kind = kind
    self.displayName = displayName
    self.url = url
    self.duration = max(0, duration)
  }
}
