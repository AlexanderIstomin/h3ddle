import Foundation
import Testing

@testable import H3ddleCore

@Suite("Project package store")
struct ProjectPackageStoreTests {
  @Test("Create, save, and load restore the timeline and copy media")
  func roundTripCopiesMedia() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleProjectStore-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ProjectPackageStore(rootURL: root)

    let source = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3-source-\(UUID().uuidString).txt")
    try Data("clip".utf8).write(to: source)
    defer { try? FileManager.default.removeItem(at: source) }
    var project = H3ddleProject(name: "Kept")
    let asset = AssetReference(
      kind: .video,
      displayName: "Opening",
      url: source,
      duration: 4
    )
    project.addAsset(asset)
    try project.timeline.appendVisual(asset)

    let created = try store.save(
      project,
      session: ProjectSession(id: project.id, packageURL: store.packageURL(for: project.id))
    )
    #expect(created.session.revision == 1)
    #expect(created.project.assets[0].url.path.contains("/Media/"))
    #expect(FileManager.default.fileExists(atPath: created.project.assets[0].url.path))

    let loaded = try store.load(id: project.id)
    #expect(loaded.project.name == "Kept")
    #expect(loaded.project.timeline.visualItems.count == 1)
    #expect(loaded.project.timeline.visualItems[0].duration == 4)
    #expect(loaded.project.assets[0].displayName == "Opening")
    #expect(loaded.session.revision == 1)
    #expect(store.recents().map(\.name) == ["Kept"])
  }

  @Test("A missing last-opened package creates Untitled Project")
  func missingLastOpenedCreates() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleProjectStore-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ProjectPackageStore(rootURL: root)
    let opened = try store.loadLastOpenedOrCreate()
    #expect(opened.project.name == "Untitled Project")
    #expect(opened.session.revision == 1)
    let again = try store.loadLastOpenedOrCreate()
    #expect(again.project.id == opened.project.id)
  }

  @Test("A missing source cannot become a phantom project media entry")
  func missingSourceIsRejected() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleProjectStore-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ProjectPackageStore(rootURL: root)
    let created = try store.create(name: "No phantom media")
    var project = created.project
    let missing = FileManager.default.temporaryDirectory
      .appendingPathComponent("missing-\(UUID().uuidString).mp4")
    project.addAsset(
      AssetReference(kind: .video, displayName: "Missing", url: missing, duration: 1)
    )

    #expect(throws: ProjectPackageStoreError.missingMedia(missing.standardizedFileURL)) {
      _ = try store.save(project, session: created.session)
    }
    let reloaded = try store.load(id: project.id)
    #expect(reloaded.project.assets.isEmpty)
    #expect(
      (try FileManager.default.contentsOfDirectory(
        at: store.mediaDirectory(for: project.id),
        includingPropertiesForKeys: nil
      )).isEmpty
    )
  }

  @Test("Relative media still resolves after the package directory is moved")
  func relativeMediaSurvivesMove() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleProjectStore-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ProjectPackageStore(rootURL: root)
    let source = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3-still-\(UUID().uuidString).png")
    try Data([0x89, 0x50, 0x4E, 0x47]).write(to: source)
    defer { try? FileManager.default.removeItem(at: source) }
    var project = H3ddleProject(name: "Move me")
    let asset = AssetReference(
      kind: .image,
      displayName: "Still",
      url: source,
      duration: 3
    )
    project.addAsset(asset)
    _ = try store.save(
      project,
      session: ProjectSession(id: project.id, packageURL: store.packageURL(for: project.id))
    )

    let movedRoot = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleProjectStore-moved-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: movedRoot) }
    try FileManager.default.copyItem(at: root, to: movedRoot)
    let moved = ProjectPackageStore(rootURL: movedRoot)
    let loaded = try moved.load(id: project.id)
    #expect(FileManager.default.fileExists(atPath: loaded.project.assets[0].url.path))
    #expect(loaded.project.assets[0].url.path.hasPrefix(movedRoot.path))
  }
}
