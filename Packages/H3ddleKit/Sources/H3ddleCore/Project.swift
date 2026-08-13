import Foundation

public struct H3ddleProject: Identifiable, Hashable, Codable, Sendable {
  public static let currentSchemaVersion = 1

  public let id: UUID
  public var schemaVersion: Int
  public var name: String
  public var assets: [AssetReference]
  public var timeline: ProjectTimeline

  public init(
    id: UUID = UUID(),
    schemaVersion: Int = H3ddleProject.currentSchemaVersion,
    name: String = "Untitled Project",
    assets: [AssetReference] = [],
    timeline: ProjectTimeline = ProjectTimeline()
  ) {
    self.id = id
    self.schemaVersion = schemaVersion
    self.name = name
    self.assets = assets
    self.timeline = timeline
  }

  public func asset(id: AssetID) -> AssetReference? {
    assets.first { $0.id == id }
  }

  public mutating func addAsset(_ asset: AssetReference) {
    guard !assets.contains(where: { $0.id == asset.id }) else { return }
    assets.append(asset)
  }
}
