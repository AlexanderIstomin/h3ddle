import Foundation

public struct H3ddleProject: Identifiable, Hashable, Codable, Sendable {
  public static let currentSchemaVersion = 3

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

  /// Library bins list newest registrations first. `assets` stays append order
  /// so interchange and timeline references are unchanged.
  public func libraryAssets(kind: MediaKind) -> [AssetReference] {
    Array(assets.reversed().filter { $0.kind == kind })
  }

  public mutating func addAsset(_ asset: AssetReference) {
    guard !assets.contains(where: { $0.id == asset.id }) else { return }
    assets.append(asset)
  }

  public func usageCount(of id: AssetID) -> Int {
    timeline.visualItems.filter { $0.assetID == id }.count
      + timeline.audioItems.filter { $0.assetID == id }.count
  }

  public mutating func renameAsset(_ id: AssetID, to name: String) {
    guard let index = assets.firstIndex(where: { $0.id == id }) else { return }
    let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }
    assets[index].displayName = trimmed
  }

  /// Removes the asset. Timeline clips that reference it are removed when
  /// `removingClips` is true; otherwise a used asset is left alone.
  @discardableResult
  public mutating func removeAsset(_ id: AssetID, removingClips: Bool = false) -> Bool {
    if usageCount(of: id) > 0 {
      guard removingClips else { return false }
      for item in timeline.visualItems where item.assetID == id {
        timeline.removeVisual(item.id)
      }
      for item in timeline.audioItems where item.assetID == id {
        timeline.removeAudio(item.id)
      }
    }
    assets.removeAll { $0.id == id }
    return true
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
