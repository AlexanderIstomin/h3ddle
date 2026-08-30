import Foundation

public struct ProjectIndexEntry: Hashable, Codable, Sendable {
  public var id: UUID
  public var name: String
  public var modifiedAt: Date

  public init(id: UUID, name: String, modifiedAt: Date = Date()) {
    self.id = id
    self.name = name
    self.modifiedAt = modifiedAt
  }
}

public struct ProjectSession: Hashable, Sendable {
  public var id: UUID
  public var packageURL: URL
  public var revision: Int
  public var extras: [String: JSONValue]

  public init(
    id: UUID,
    packageURL: URL,
    revision: Int = 0,
    extras: [String: JSONValue] = [:]
  ) {
    self.id = id
    self.packageURL = packageURL
    self.revision = revision
    self.extras = extras
  }
}

private struct ProjectIndex: Codable, Sendable {
  var lastOpenedID: UUID?
  var recents: [ProjectIndexEntry]
}

/// App-managed project packages: `Projects/{id}/project.json` + `Media/`.
public struct ProjectPackageStore: Sendable {
  public static let documentName = "project.json"
  public static let mediaDirectoryName = "Media"
  public static let indexName = "index.json"

  public var rootURL: URL

  public init(
    rootURL: URL = URL.applicationSupportDirectory
      .appendingPathComponent("H3ddle", isDirectory: true)
      .appendingPathComponent("Projects", isDirectory: true)
  ) {
    self.rootURL = rootURL
  }

  public func packageURL(for id: UUID) -> URL {
    rootURL.appendingPathComponent(id.uuidString.lowercased(), isDirectory: true)
  }

  public func documentURL(for id: UUID) -> URL {
    packageURL(for: id).appendingPathComponent(Self.documentName)
  }

  public func mediaDirectory(for id: UUID) -> URL {
    packageURL(for: id).appendingPathComponent(Self.mediaDirectoryName, isDirectory: true)
  }

  public func create(name: String = "Untitled Project") throws -> (
    project: H3ddleProject, session: ProjectSession
  ) {
    let project = H3ddleProject(name: name)
    try preparePackage(project.id)
    let session = ProjectSession(
      id: project.id,
      packageURL: packageURL(for: project.id)
    )
    return try save(project, session: session)
  }

  public func load(id: UUID) throws -> (project: H3ddleProject, session: ProjectSession) {
    let data = try Data(contentsOf: documentURL(for: id))
    let document = try InterchangeProjection.decodeDocument(data)
    let mediaRoot = mediaDirectory(for: id)
    let project = try InterchangeProjection.project(from: document) { asset in
      resolveMedia(asset.src, mediaRoot: mediaRoot)
    }
    let session = ProjectSession(
      id: project.id,
      packageURL: packageURL(for: id),
      revision: document.revision,
      extras: document.extras
    )
    recordOpen(project)
    return (project, session)
  }

  public func loadLastOpenedOrCreate() throws -> (project: H3ddleProject, session: ProjectSession)
  {
    if let id = index().lastOpenedID, FileManager.default.fileExists(atPath: documentURL(for: id).path)
    {
      return try load(id: id)
    }
    return try create()
  }

  @discardableResult
  public func save(
    _ project: H3ddleProject,
    session: ProjectSession
  ) throws -> (project: H3ddleProject, session: ProjectSession) {
    try preparePackage(project.id)
    var adopted = project
    try adoptMedia(into: &adopted, packageID: project.id)
    let revision = session.revision + 1
    let document = try InterchangeProjection.document(
      from: adopted,
      revision: revision,
      extras: session.extras,
      mediaLocator: { asset in
        relativeMediaPath(for: asset, packageID: project.id)
      }
    )
    let data = try InterchangeProjection.encode(document)
    try data.write(to: documentURL(for: project.id), options: .atomic)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o600],
      ofItemAtPath: documentURL(for: project.id).path
    )
    recordOpen(adopted)
    return (
      adopted,
      ProjectSession(
        id: adopted.id,
        packageURL: packageURL(for: adopted.id),
        revision: revision,
        extras: document.extras
      )
    )
  }

  public func recents() -> [ProjectIndexEntry] {
    index().recents
  }

  public func delete(id: UUID) throws {
    try FileManager.default.removeItem(at: packageURL(for: id))
    var stored = index()
    stored.recents.removeAll { $0.id == id }
    if stored.lastOpenedID == id { stored.lastOpenedID = stored.recents.first?.id }
    try writeIndex(stored)
  }

  private func preparePackage(_ id: UUID) throws {
    try FileManager.default.createDirectory(
      at: mediaDirectory(for: id),
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
  }

  private func adoptMedia(into project: inout H3ddleProject, packageID: UUID) throws {
    let mediaRoot = mediaDirectory(for: packageID)
    var next: [AssetReference] = []
    for asset in project.assets {
      var adopted = asset
      let destination = mediaFileURL(for: asset, in: mediaRoot)
      let source = asset.url.standardizedFileURL
      if source != destination.standardizedFileURL {
        if FileManager.default.fileExists(atPath: source.path) {
          if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
          }
          try FileManager.default.copyItem(at: source, to: destination)
        }
      }
      adopted.url = destination
      next.append(adopted)
    }
    project.assets = next
  }

  private func mediaFileURL(for asset: AssetReference, in mediaRoot: URL) -> URL {
    let ext = asset.url.pathExtension
    let name =
      ext.isEmpty
      ? asset.id.rawValue.uuidString.lowercased()
      : "\(asset.id.rawValue.uuidString.lowercased()).\(ext)"
    return mediaRoot.appendingPathComponent(name)
  }

  private func relativeMediaPath(for asset: AssetReference, packageID: UUID) -> String {
    "\(Self.mediaDirectoryName)/\(mediaFileURL(for: asset, in: mediaDirectory(for: packageID)).lastPathComponent)"
  }

  private func resolveMedia(_ src: String, mediaRoot: URL) -> URL {
    if src.hasPrefix("Media/") || src.hasPrefix("\(Self.mediaDirectoryName)/") {
      return mediaRoot.appendingPathComponent((src as NSString).lastPathComponent)
    }
    if let url = URL(string: src), url.scheme != nil {
      return url
    }
    return mediaRoot.appendingPathComponent((src as NSString).lastPathComponent)
  }

  private func indexURL() -> URL {
    rootURL.appendingPathComponent(Self.indexName)
  }

  private func index() -> ProjectIndex {
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    guard let data = try? Data(contentsOf: indexURL()),
      let stored = try? decoder.decode(ProjectIndex.self, from: data)
    else {
      return ProjectIndex(lastOpenedID: nil, recents: [])
    }
    return stored
  }

  private func writeIndex(_ stored: ProjectIndex) throws {
    try FileManager.default.createDirectory(
      at: rootURL,
      withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700]
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    encoder.dateEncodingStrategy = .iso8601
    let data = try encoder.encode(stored)
    try data.write(to: indexURL(), options: .atomic)
  }

  private func recordOpen(_ project: H3ddleProject) {
    var stored = index()
    stored.lastOpenedID = project.id
    let entry = ProjectIndexEntry(id: project.id, name: project.name)
    stored.recents.removeAll { $0.id == project.id }
    stored.recents.insert(entry, at: 0)
    if stored.recents.count > 20 {
      stored.recents.removeLast(stored.recents.count - 20)
    }
    try? writeIndex(stored)
  }
}
