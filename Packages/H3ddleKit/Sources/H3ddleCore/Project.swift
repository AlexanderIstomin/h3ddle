import Foundation

public struct H3ddleProject: Identifiable, Hashable, Codable, Sendable {
  public static let currentSchemaVersion = 1

  public let id: UUID
  public var schemaVersion: Int
  public var name: String
  public var assets: [AssetReference]
  public var timeline: ProjectTimeline
  public var settings: ProjectSettings

  public init(
    id: UUID = UUID(),
    schemaVersion: Int = H3ddleProject.currentSchemaVersion,
    name: String = "Untitled Project",
    assets: [AssetReference] = [],
    timeline: ProjectTimeline = ProjectTimeline(),
    settings: ProjectSettings = .default
  ) {
    self.id = id
    self.schemaVersion = schemaVersion
    self.name = name
    self.assets = assets
    self.timeline = timeline
    self.settings = settings
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    id = try container.decode(UUID.self, forKey: .id)
    schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
    name = try container.decode(String.self, forKey: .name)
    assets = try container.decode([AssetReference].self, forKey: .assets)
    timeline = try container.decode(ProjectTimeline.self, forKey: .timeline)
    settings = try container.decodeIfPresent(ProjectSettings.self, forKey: .settings) ?? .default
  }

  public func asset(id: AssetID) -> AssetReference? {
    assets.first { $0.id == id }
  }

  public mutating func addAsset(_ asset: AssetReference) {
    guard !assets.contains(where: { $0.id == asset.id }) else { return }
    assets.append(asset)
  }

  enum CodingKeys: String, CodingKey {
    case id
    case schemaVersion
    case name
    case assets
    case timeline
    case settings
  }
}
