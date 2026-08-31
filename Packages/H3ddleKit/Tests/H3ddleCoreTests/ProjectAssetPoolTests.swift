import Foundation
import Testing

@testable import H3ddleCore

@Suite("Project asset pool")
struct ProjectAssetPoolTests {
  @Test("An unused asset can be removed; a used one cannot")
  func removeUnusedOnly() throws {
    var project = H3ddleProject()
    let used = AssetReference(
      kind: .video,
      displayName: "On timeline",
      url: URL(fileURLWithPath: "/tmp/used.mp4"),
      duration: 4
    )
    let spare = AssetReference(
      kind: .image,
      displayName: "Library only",
      url: URL(fileURLWithPath: "/tmp/spare.png"),
      duration: 3
    )
    project.addAsset(used)
    project.addAsset(spare)
    try project.timeline.appendVisual(used)
    #expect(project.usageCount(of: used.id) == 1)
    #expect(project.usageCount(of: spare.id) == 0)
    let blocked = project.removeAsset(used.id)
    #expect(!blocked)
    #expect(project.asset(id: used.id) != nil)
    let removed = project.removeAsset(spare.id)
    #expect(removed)
    #expect(project.asset(id: spare.id) == nil)
  }

  @Test("Removing an asset with clips clears those clips")
  func removeWithClips() throws {
    var project = H3ddleProject()
    let video = AssetReference(
      kind: .video,
      displayName: "Shot",
      url: URL(fileURLWithPath: "/tmp/shot.mp4"),
      duration: 4
    )
    project.addAsset(video)
    try project.timeline.appendVisual(video)
    try project.timeline.appendVisual(video)
    let removed = project.removeAsset(video.id, removingClips: true)
    #expect(removed)
    #expect(project.timeline.visualItems.isEmpty)
    #expect(project.assets.isEmpty)
  }

  @Test("Library bins list newest assets first")
  func libraryAssetsAreNewestFirst() {
    var project = H3ddleProject()
    let first = AssetReference(
      kind: .video,
      displayName: "Older",
      url: URL(fileURLWithPath: "/tmp/older.mp4"),
      duration: 4
    )
    let second = AssetReference(
      kind: .video,
      displayName: "Newer",
      url: URL(fileURLWithPath: "/tmp/newer.mp4"),
      duration: 4
    )
    let audio = AssetReference(
      kind: .audio,
      displayName: "Bed",
      url: URL(fileURLWithPath: "/tmp/bed.wav"),
      duration: 8
    )
    project.addAsset(first)
    project.addAsset(second)
    project.addAsset(audio)
    #expect(project.libraryAssets(kind: .video).map(\.id) == [second.id, first.id])
    #expect(project.libraryAssets(kind: .audio).map(\.id) == [audio.id])
    #expect(project.assets.map(\.id) == [first.id, second.id, audio.id])
  }

  @Test("Generation inputs persist without appearing in Media bins")
  func generationInputsStayHidden() {
    var project = H3ddleProject()
    let visible = AssetReference(
      kind: .image,
      displayName: "Visible",
      url: URL(fileURLWithPath: "/tmp/visible.png"),
      duration: 3
    )
    let dependency = AssetReference(
      kind: .image,
      displayName: "Reference",
      url: URL(fileURLWithPath: "/tmp/reference.png"),
      duration: 3,
      metadata: [AssetMetadataKey.generationInput: .bool(true)]
    )

    project.addAsset(visible)
    project.addAsset(dependency)

    #expect(project.libraryAssets(kind: .image).map(\.id) == [visible.id])
    #expect(project.asset(id: dependency.id) == dependency)
  }

  @Test("Registering twice does not duplicate")
  func addAssetIsIdempotent() {
    var project = H3ddleProject()
    let asset = AssetReference(
      kind: .audio,
      displayName: "Bed",
      url: URL(fileURLWithPath: "/tmp/bed.wav"),
      duration: 8
    )
    project.addAsset(asset)
    project.addAsset(asset)
    #expect(project.assets.count == 1)
  }
}
