import Foundation
import Testing

@testable import H3ddleCore

@Suite("Two-track timeline")
struct TimelineTests {
  @Test("Visual assets append in program order")
  func appendsVisualAssets() throws {
    var timeline = ProjectTimeline()
    let video = AssetReference(
      kind: .video,
      displayName: "Opening",
      url: URL(fileURLWithPath: "/tmp/opening.mp4"),
      duration: 5
    )
    let image = AssetReference(
      kind: .image,
      displayName: "Still",
      url: URL(fileURLWithPath: "/tmp/still.png"),
      duration: 3
    )

    try timeline.appendVisual(video)
    try timeline.appendVisual(image)

    #expect(timeline.visualItems.map(\.assetID) == [video.id, image.id])
    #expect(timeline.visualDuration == 8)
  }

  @Test("Audio appends at the current audio end")
  func appendsAudioAtTrackEnd() throws {
    var timeline = ProjectTimeline()
    let first = audioAsset(name: "Dialogue", duration: 4)
    let second = audioAsset(name: "Music", duration: 6)

    let firstItem = try timeline.appendAudio(first)
    let secondItem = try timeline.appendAudio(second)

    #expect(firstItem.startTime == 0)
    #expect(secondItem.startTime == 4)
    #expect(timeline.audioTrackEnd == 10)
  }

  @Test("Removing audio preserves later synchronization")
  func removingAudioPreservesLaterStartTimes() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendAudio(audioAsset(name: "One", duration: 4))
    let second = try timeline.appendAudio(audioAsset(name: "Two", duration: 6))

    timeline.removeAudio(first.id)

    #expect(timeline.audioItems.map(\.id) == [second.id])
    #expect(timeline.audioItems[0].startTime == 4)
  }

  @Test("Trailing audio is measured against visual duration")
  func reportsTrailingAudio() throws {
    var timeline = ProjectTimeline()
    try timeline.appendVisual(
      AssetReference(
        kind: .video,
        displayName: "Visual",
        url: URL(fileURLWithPath: "/tmp/visual.mp4"),
        duration: 5
      )
    )
    try timeline.appendAudio(audioAsset(name: "Long mix", duration: 7.5))

    #expect(timeline.trailingAudioDuration == 2.5)
  }

  @Test("Audio cannot enter the visual track")
  func rejectsWrongAssetKind() {
    var timeline = ProjectTimeline()

    #expect(throws: TimelineError.expectedVisualAsset) {
      try timeline.appendVisual(audioAsset(name: "Audio", duration: 2))
    }
  }

  private func audioAsset(name: String, duration: TimeInterval) -> AssetReference {
    AssetReference(
      kind: .audio,
      displayName: name,
      url: URL(fileURLWithPath: "/tmp/\(name).m4a"),
      duration: duration
    )
  }
}
