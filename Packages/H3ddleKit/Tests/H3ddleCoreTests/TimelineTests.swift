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
    #expect(timeline.visualItems[0].includesNativeAudio)
    #expect(!timeline.visualItems[1].includesNativeAudio)
  }

  @Test("Split at playhead divides a visual clip and carries the source offset")
  func splitsVisualAtPlayhead() throws {
    var timeline = ProjectTimeline()
    let video = videoAsset(name: "Shot", duration: 6)
    let later = videoAsset(name: "Next", duration: 4)
    try timeline.appendVisual(video)
    try timeline.appendVisual(later)
    let first = timeline.visualItems[0]
    timeline.setVisualTrim(
      first.id,
      VisualTrim(duration: 6, sourceOffset: 1, gapBefore: 0)
    )

    #expect(!timeline.canSplitVisual(first.id, at: 0, framesPerSecond: 24))
    #expect(!timeline.canSplitVisual(first.id, at: 6, framesPerSecond: 24))
    #expect(timeline.canSplitVisual(first.id, at: 2.5, framesPerSecond: 24))

    let split = timeline.splitVisual(
      first.id,
      at: 2.5,
      sourceKind: .video,
      framesPerSecond: 24
    )
    #expect(split?.left == first.id)
    #expect(timeline.visualItems.count == 3)
    #expect(abs(timeline.visualItems[0].duration - 2.5) < 0.000_1)
    #expect(abs(timeline.visualItems[0].sourceOffset - 1) < 0.000_1)
    #expect(abs(timeline.visualItems[1].duration - 3.5) < 0.000_1)
    #expect(abs(timeline.visualItems[1].sourceOffset - 3.5) < 0.000_1)
    #expect(timeline.visualItems[1].gapBefore == 0)
    #expect(timeline.visualItems[2].assetID == later.id)
    #expect(abs(timeline.visualDuration - 10) < 0.000_1)
  }

  @Test("An image split does not invent a source offset")
  func splitsImageWithoutSourceOffset() throws {
    var timeline = ProjectTimeline()
    let image = AssetReference(
      kind: .image,
      displayName: "Still",
      url: URL(fileURLWithPath: "/tmp/still.png"),
      duration: 4
    )
    try timeline.appendVisual(image)
    _ = timeline.splitVisual(
      timeline.visualItems[0].id,
      at: 1.5,
      sourceKind: .image,
      framesPerSecond: 24
    )
    #expect(timeline.visualItems[0].sourceOffset == 0)
    #expect(timeline.visualItems[1].sourceOffset == 0)
    #expect(abs(timeline.visualItems[1].duration - 2.5) < 0.000_1)
  }

  @Test("Audio split keeps later start times and advances the right-hand in-point")
  func splitsAudioAtPlayhead() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendAudio(audioAsset(name: "One", duration: 5))
    let second = try timeline.appendAudio(audioAsset(name: "Two", duration: 3))
    #expect(timeline.canSplitAudio(first.id, at: 2, framesPerSecond: 24))
    #expect(!timeline.canSplitAudio(first.id, at: 0, framesPerSecond: 24))

    let split = timeline.splitAudio(first.id, at: 2, framesPerSecond: 24)
    #expect(split?.left == first.id)
    #expect(timeline.audioItems.count == 3)
    #expect(timeline.audioItems[0].startTime == 0)
    #expect(abs(timeline.audioItems[0].duration - 2) < 0.000_1)
    #expect(abs(timeline.audioItems[1].startTime - 2) < 0.000_1)
    #expect(abs(timeline.audioItems[1].duration - 3) < 0.000_1)
    #expect(abs(timeline.audioItems[1].sourceOffset - 2) < 0.000_1)
    #expect(timeline.audioItems[2].id == second.id)
    #expect(timeline.audioItems[2].startTime == 5)
  }

  @Test("Imported video can append without a native soundtrack")
  func importedVideoWithoutNativeAudio() throws {
    var timeline = ProjectTimeline()
    let video = videoAsset(name: "Silent", duration: 4)
    let item = try timeline.appendVisual(video, includesNativeAudio: false)
    #expect(!item.includesNativeAudio)
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

  @Test("Trailing visual trim ripples later clips and keeps video inside the source")
  func trailingTrimRipplesAndClampsVideo() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendVisual(videoAsset(name: "One", duration: 5))
    try timeline.appendVisual(videoAsset(name: "Two", duration: 4))

    let shorter = VisualTrimMath.apply(
      edge: .trailing,
      delta: -2,
      startTime: 0,
      duration: 5,
      sourceOffset: 0,
      gapBefore: 0,
      sourceLimit: 5,
      minimumDuration: 1 / 24
    )
    timeline.setVisualTrim(first.id, shorter)
    #expect(abs(timeline.visualItems[0].duration - 3) < 0.000_1)
    #expect(abs(timeline.visualPlacements[1].startTime - 3) < 0.000_1)

    let overstretched = VisualTrimMath.apply(
      edge: .trailing,
      delta: 20,
      startTime: 0,
      duration: 3,
      sourceOffset: 0,
      gapBefore: 0,
      sourceLimit: 5,
      minimumDuration: 1 / 24
    )
    #expect(abs(overstretched.duration - 5) < 0.000_1)
  }

  @Test("Trim cannot collapse a clip below one frame")
  func trimKeepsAMinimumDuration() {
    let trim = VisualTrimMath.apply(
      edge: .trailing,
      delta: -10,
      startTime: 0,
      duration: 3,
      sourceOffset: 0,
      gapBefore: 0,
      sourceLimit: 3,
      minimumDuration: 1 / 24
    )
    #expect(abs(trim.duration - 1 / 24) < 0.000_1)
  }

  @Test("Images can extend past the generated still length")
  func imageTrimIsUnbounded() {
    let trim = VisualTrimMath.apply(
      edge: .trailing,
      delta: 7,
      startTime: 0,
      duration: 3,
      sourceOffset: 0,
      gapBefore: 0,
      sourceLimit: nil,
      minimumDuration: 1 / 24
    )
    #expect(abs(trim.duration - 10) < 0.000_1)
  }

  @Test("Leading visual trim keeps the out point and updates the source in-point")
  func leadingTrimPreservesOutPoint() throws {
    var timeline = ProjectTimeline()
    try timeline.appendVisual(videoAsset(name: "One", duration: 5))
    let second = try timeline.appendVisual(videoAsset(name: "Two", duration: 4))

    let trim = VisualTrimMath.apply(
      edge: .leading,
      delta: 1,
      startTime: 5,
      duration: 4,
      sourceOffset: 0,
      gapBefore: 0,
      sourceLimit: 4,
      minimumDuration: 1 / 24
    )
    timeline.setVisualTrim(second.id, trim)

    #expect(abs(timeline.visualItems[1].gapBefore - 1) < 0.000_1)
    #expect(abs(timeline.visualItems[1].duration - 3) < 0.000_1)
    #expect(abs(timeline.visualItems[1].sourceOffset - 1) < 0.000_1)
    #expect(abs(timeline.visualPlacements[1].startTime - 6) < 0.000_1)
    #expect(abs(timeline.visualDuration - 9) < 0.000_1)
  }

  @Test("Leading trim cannot enter the previous clip or the source head")
  func leadingTrimClamps() {
    let blockedByNeighbor = VisualTrimMath.apply(
      edge: .leading,
      delta: -2,
      startTime: 5,
      duration: 4,
      sourceOffset: 0,
      gapBefore: 0,
      sourceLimit: 4,
      minimumDuration: 1 / 24
    )
    #expect(abs(blockedByNeighbor.gapBefore) < 0.000_1)
    #expect(abs(blockedByNeighbor.duration - 4) < 0.000_1)

    let blockedBySource = VisualTrimMath.apply(
      edge: .leading,
      delta: -2,
      startTime: 2,
      duration: 3,
      sourceOffset: 0.5,
      gapBefore: 2,
      sourceLimit: 5,
      minimumDuration: 1 / 24
    )
    #expect(abs(blockedBySource.sourceOffset) < 0.000_1)
    #expect(abs(blockedBySource.duration - 3.5) < 0.000_1)
    #expect(abs(blockedBySource.gapBefore - 1.5) < 0.000_1)
  }

  @Test("Trailing audio trim keeps later start times and stays in the source")
  func trailingAudioTrimDoesNotRipple() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendAudio(audioAsset(name: "One", duration: 4))
    let second = try timeline.appendAudio(audioAsset(name: "Two", duration: 6))
    let bounds = timeline.audioNeighborBounds(of: first.id)

    let shorter = AudioTrimMath.apply(
      edge: .trailing,
      delta: -1.5,
      startTime: 0,
      duration: 4,
      sourceOffset: 0,
      sourceLimit: 4,
      earliestStart: bounds.earliestStart,
      latestEnd: bounds.latestEnd,
      minimumDuration: 1 / 24
    )
    timeline.setAudioTrim(first.id, shorter)
    #expect(abs(timeline.audioItems[0].duration - 2.5) < 0.000_1)
    #expect(timeline.audioItems[1].startTime == second.startTime)

    let blockedByNeighbor = AudioTrimMath.apply(
      edge: .trailing,
      delta: 8,
      startTime: 0,
      duration: 2.5,
      sourceOffset: 0,
      sourceLimit: 8,
      earliestStart: 0,
      latestEnd: 4,
      minimumDuration: 1 / 24
    )
    #expect(abs(blockedByNeighbor.duration - 4) < 0.000_1)
  }

  @Test("Leading audio trim keeps the out point and updates the source in-point")
  func leadingAudioTrimPreservesOutPoint() throws {
    var timeline = ProjectTimeline()
    try timeline.appendAudio(audioAsset(name: "One", duration: 4))
    let second = try timeline.appendAudio(audioAsset(name: "Two", duration: 6))
    let bounds = timeline.audioNeighborBounds(of: second.id)

    let trim = AudioTrimMath.apply(
      edge: .leading,
      delta: 1,
      startTime: 4,
      duration: 6,
      sourceOffset: 0,
      sourceLimit: 6,
      earliestStart: bounds.earliestStart,
      latestEnd: bounds.latestEnd,
      minimumDuration: 1 / 24
    )
    timeline.setAudioTrim(second.id, trim)
    #expect(abs(timeline.audioItems[1].startTime - 5) < 0.000_1)
    #expect(abs(timeline.audioItems[1].duration - 5) < 0.000_1)
    #expect(abs(timeline.audioItems[1].sourceOffset - 1) < 0.000_1)
    #expect(abs(timeline.audioItems[1].endTime - 10) < 0.000_1)
    #expect(timeline.audioItems[0].startTime == 0)
  }

  @Test("Legacy audio items decode without a source offset")
  func decodesLegacyAudioItems() throws {
    let item = AudioItem(
      assetID: AssetID(),
      startTime: 2,
      duration: 4
    )
    var encoded = try JSONEncoder().encode(item)
    var object = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
    object.removeValue(forKey: "sourceOffset")
    encoded = try JSONSerialization.data(withJSONObject: object)
    let decoded = try JSONDecoder().decode(AudioItem.self, from: encoded)
    #expect(decoded.sourceOffset == 0)
    #expect(decoded.startTime == 2)
    #expect(decoded.duration == 4)
  }

  @Test("Legacy visual items decode without trim fields")
  func decodesLegacyVisualItems() throws {
    let item = VisualItem(
      assetID: AssetID(),
      duration: 4,
      includesNativeAudio: true
    )
    var encoded = try JSONEncoder().encode(item)
    var object = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
    object.removeValue(forKey: "sourceOffset")
    object.removeValue(forKey: "gapBefore")
    encoded = try JSONSerialization.data(withJSONObject: object)
    let decoded = try JSONDecoder().decode(VisualItem.self, from: encoded)
    #expect(decoded.sourceOffset == 0)
    #expect(decoded.gapBefore == 0)
    #expect(decoded.duration == 4)
    #expect(decoded.canvasFit == .fit)
    #expect(decoded.rotationTurns == 0)
  }

  @Test("Visual canvas fit and rotation persist and wrap")
  func visualCanvasTransformPersists() throws {
    var timeline = ProjectTimeline()
    let video = videoAsset(name: "Shot", duration: 4)
    let item = try timeline.appendVisual(video)
    #expect(item.canvasFit == .fit)
    #expect(item.rotationTurns == 0)

    timeline.setVisualCanvasFit(item.id, .cover)
    timeline.rotateVisual(item.id)
    timeline.rotateVisual(item.id)
    #expect(timeline.visualItems[0].canvasFit == .cover)
    #expect(timeline.visualItems[0].rotationTurns == 2)

    timeline.rotateVisual(item.id, by: 3)
    #expect(timeline.visualItems[0].rotationTurns == 1)

    let encoded = try JSONEncoder().encode(timeline.visualItems[0])
    let decoded = try JSONDecoder().decode(VisualItem.self, from: encoded)
    #expect(decoded.canvasFit == .cover)
    #expect(decoded.rotationTurns == 1)

    _ = timeline.splitVisual(item.id, at: 2, sourceKind: .video, framesPerSecond: 24)
    #expect(timeline.visualItems[0].canvasFit == .cover)
    #expect(timeline.visualItems[1].canvasFit == .cover)
    #expect(timeline.visualItems[0].rotationTurns == 1)
    #expect(timeline.visualItems[1].rotationTurns == 1)
  }

  @Test("Legacy visual items decode without canvas fields")
  func decodesLegacyVisualCanvasFields() throws {
    let item = VisualItem(
      assetID: AssetID(),
      duration: 4,
      includesNativeAudio: true,
      canvasFit: .cover,
      rotationTurns: 2
    )
    var encoded = try JSONEncoder().encode(item)
    var object = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
    object.removeValue(forKey: "canvasFit")
    object.removeValue(forKey: "rotationTurns")
    encoded = try JSONSerialization.data(withJSONObject: object)
    let decoded = try JSONDecoder().decode(VisualItem.self, from: encoded)
    #expect(decoded.canvasFit == .fit)
    #expect(decoded.rotationTurns == 0)
  }

  private func videoAsset(name: String, duration: TimeInterval) -> AssetReference {
    AssetReference(
      kind: .video,
      displayName: name,
      url: URL(fileURLWithPath: "/tmp/\(name).mp4"),
      duration: duration
    )
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
