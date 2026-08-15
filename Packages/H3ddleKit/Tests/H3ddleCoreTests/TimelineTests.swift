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
    #expect(decoded.transition == nil)
  }

  @Test("A transition lives on an adjacent incoming cut and is clamped")
  func visualTransitionOnAdjacentCut() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendVisual(videoAsset(name: "A", duration: 2))
    let second = try timeline.appendVisual(videoAsset(name: "B", duration: 1))
    #expect(!timeline.canApplyVisualTransition(first.id))
    #expect(timeline.canApplyVisualTransition(second.id))
    #expect(timeline.maximumVisualTransitionDuration(of: second.id) == 1)

    timeline.setVisualTransition(
      second.id,
      VisualTransition(kind: .wipe, duration: 4)
    )
    #expect(timeline.visualItems[1].transition?.kind == .wipe)
    #expect(abs((timeline.visualItems[1].transition?.duration ?? 0) - 1) < 0.000_1)
    #expect(abs(timeline.transitionOverlap(of: second.id) - 1) < 0.000_1)
    #expect(abs(timeline.visualPlacements[1].startTime - 1) < 0.000_1)
    #expect(abs(timeline.visualDuration - 2) < 0.000_1)

    timeline.setVisualTrim(
      second.id,
      VisualTrim(duration: 1, sourceOffset: 0, gapBefore: 0.2)
    )
    #expect(!timeline.canApplyVisualTransition(second.id))
    #expect(timeline.transitionOverlap(of: second.id) == 0)

    timeline.setVisualTrim(second.id, VisualTrim(duration: 1, sourceOffset: 0, gapBefore: 0))
    _ = timeline.splitVisual(second.id, at: 1.5, sourceKind: .video, framesPerSecond: 24)
    #expect(timeline.visualItems[1].transition?.kind == .wipe)
    #expect(timeline.visualItems[2].transition == nil)
  }

  @Test("A transition overlaps the incoming clip into the outgoing tail")
  func transitionOverlapsIncomingIntoOutgoing() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendVisual(videoAsset(name: "A", duration: 4))
    let second = try timeline.appendVisual(videoAsset(name: "B", duration: 4))
    timeline.setVisualTransition(second.id, VisualTransition(kind: .dissolve, duration: 1))
    #expect(abs(timeline.visualPlacements[0].startTime) < 0.000_1)
    #expect(abs(timeline.visualPlacements[1].startTime - 3) < 0.000_1)
    #expect(abs(timeline.visualDuration - 7) < 0.000_1)
    #expect(timeline.visualItems[0].id == first.id)
    timeline.setVisualTransition(second.id, nil)
    #expect(abs(timeline.visualPlacements[1].startTime - 4) < 0.000_1)
    #expect(abs(timeline.visualDuration - 8) < 0.000_1)
  }

  @Test("Legacy visual items decode without a transition")
  func decodesLegacyVisualTransition() throws {
    let item = VisualItem(
      assetID: AssetID(),
      duration: 3,
      includesNativeAudio: true,
      transition: VisualTransition(kind: .dissolve, duration: 0.4)
    )
    var encoded = try JSONEncoder().encode(item)
    var object = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
    object.removeValue(forKey: "transition")
    encoded = try JSONSerialization.data(withJSONObject: object)
    let decoded = try JSONDecoder().decode(VisualItem.self, from: encoded)
    #expect(decoded.transition == nil)
    #expect(decoded.effects.isEmpty)
  }

  @Test("Visual effects append, persist, and copy on split")
  func visualEffectsCopyOnSplit() throws {
    var timeline = ProjectTimeline()
    let video = videoAsset(name: "Shot", duration: 4)
    let item = try timeline.appendVisual(video)
    let added = timeline.addVisualEffect(item.id, kind: .vignette)
    #expect(added?.kind == .vignette)
    #expect(timeline.visualItems[0].effects.count == 1)

    _ = timeline.splitVisual(item.id, at: 2, sourceKind: .video, framesPerSecond: 24)
    #expect(timeline.visualItems[0].effects.count == 1)
    #expect(timeline.visualItems[1].effects.count == 1)
    #expect(timeline.visualItems[0].effects[0].id != timeline.visualItems[1].effects[0].id)
    #expect(timeline.visualItems[1].effects[0].kind == .vignette)

    timeline.removeVisualEffect(item.id, effectID: timeline.visualItems[0].effects[0].id)
    #expect(timeline.visualItems[0].effects.isEmpty)
  }

  @Test("Legacy visual items decode without effects")
  func decodesLegacyVisualEffects() throws {
    let item = VisualItem(
      assetID: AssetID(),
      duration: 3,
      includesNativeAudio: true,
      effects: [VisualEffectInstance(kind: .blur)]
    )
    var encoded = try JSONEncoder().encode(item)
    var object = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
    object.removeValue(forKey: "effects")
    encoded = try JSONSerialization.data(withJSONObject: object)
    let decoded = try JSONDecoder().decode(VisualItem.self, from: encoded)
    #expect(decoded.effects.isEmpty)
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

  @Test("Duplicate inserts a visual copy after the original")
  func duplicatesVisualAfterOriginal() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendVisual(videoAsset(name: "Shot", duration: 4))
    try timeline.appendVisual(videoAsset(name: "Next", duration: 3))
    timeline.setVisualCanvasFit(first.id, .cover)
    timeline.rotateVisual(first.id)
    timeline.setVisualTrim(
      first.id,
      VisualTrim(duration: 3.5, sourceOffset: 0.5, gapBefore: 0)
    )
    _ = timeline.addVisualEffect(first.id, kind: .vignette)
    timeline.setVisualTransition(
      timeline.visualItems[1].id,
      VisualTransition(kind: .dissolve, duration: 0.4)
    )

    let copy = timeline.duplicateVisual(first.id)
    #expect(copy != nil)
    #expect(timeline.visualItems.count == 3)
    #expect(timeline.visualItems[0].id == first.id)
    #expect(timeline.visualItems[1].id == copy?.id)
    #expect(timeline.visualItems[1].assetID == first.assetID)
    #expect(timeline.visualItems[1].id != first.id)
    #expect(abs(timeline.visualItems[1].duration - 3.5) < 0.000_1)
    #expect(abs(timeline.visualItems[1].sourceOffset - 0.5) < 0.000_1)
    #expect(timeline.visualItems[1].gapBefore == 0)
    #expect(timeline.visualItems[1].canvasFit == .cover)
    #expect(timeline.visualItems[1].rotationTurns == 1)
    #expect(timeline.visualItems[1].includesNativeAudio)
    #expect(timeline.visualItems[1].transition == nil)
    #expect(timeline.visualItems[1].effects.count == 1)
    #expect(timeline.visualItems[1].effects[0].id != timeline.visualItems[0].effects[0].id)
    #expect(timeline.visualItems[2].transition?.kind == .dissolve)
    #expect(abs(timeline.visualDuration - 9.6) < 0.000_1)
  }

  @Test("Visual reorder shuffles order and drops stale transitions")
  func reordersVisualItems() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendVisual(videoAsset(name: "A", duration: 2))
    let second = try timeline.appendVisual(videoAsset(name: "B", duration: 3))
    let third = try timeline.appendVisual(videoAsset(name: "C", duration: 4))
    timeline.setVisualTransition(second.id, VisualTransition(kind: .wipe, duration: 0.5))
    timeline.setVisualTransition(third.id, VisualTransition(kind: .fade, duration: 0.5))

    timeline.moveVisual(first.id, toIndex: 2)
    #expect(timeline.visualItems.map(\.id) == [second.id, third.id, first.id])
    #expect(timeline.visualItems[0].transition == nil)
    #expect(timeline.visualItems[1].transition?.kind == .fade)
    #expect(timeline.visualItems[2].transition == nil)
    #expect(abs(timeline.visualDuration - 8.5) < 0.000_1)

    timeline.moveVisual(first.id, toIndex: 0)
    #expect(timeline.visualItems.map(\.id) == [first.id, second.id, third.id])

    timeline.moveVisual(first.id, toIndex: 0)
    #expect(timeline.visualItems.map(\.id) == [first.id, second.id, third.id])
  }

  @Test("Removing the outgoing visual drops the incoming transition")
  func removeVisualDropsFollowerTransition() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendVisual(videoAsset(name: "A", duration: 2))
    let second = try timeline.appendVisual(videoAsset(name: "B", duration: 2))
    timeline.setVisualTransition(second.id, VisualTransition(kind: .dissolve, duration: 0.5))
    timeline.removeVisual(first.id)
    #expect(timeline.visualItems.map(\.id) == [second.id])
    #expect(timeline.visualItems[0].transition == nil)
  }

  @Test("Duplicate audio sits after the source and only ripples when needed")
  func duplicatesAudioAfterSource() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendAudio(audioAsset(name: "One", duration: 4))
    let packed = try timeline.appendAudio(audioAsset(name: "Two", duration: 3))
    let copy = timeline.duplicateAudio(first.id)
    #expect(copy != nil)
    #expect(timeline.audioItems[0].id == first.id)
    #expect(timeline.audioItems[1].id == copy?.id)
    #expect(abs(timeline.audioItems[1].startTime - 4) < 0.000_1)
    #expect(abs(timeline.audioItems[1].duration - 4) < 0.000_1)
    #expect(timeline.audioItems[1].sourceOffset == 0)
    #expect(abs(timeline.audioItems[2].startTime - 8) < 0.000_1)
    #expect(timeline.audioItems[2].id == packed.id)

    var gapped = ProjectTimeline()
    let early = try gapped.appendAudio(audioAsset(name: "Early", duration: 2))
    let late = try gapped.appendAudio(audioAsset(name: "Late", duration: 2))
    gapped.setAudioTrim(
      late.id,
      AudioTrim(startTime: 6, duration: 2, sourceOffset: 0)
    )
    _ = gapped.duplicateAudio(early.id)
    #expect(abs(gapped.audioItems[1].startTime - 2) < 0.000_1)
    #expect(abs(gapped.audioItems[2].startTime - 6) < 0.000_1)
  }

  @Test("Audio start slides inside neighbor bounds")
  func slidesAudioInsideNeighbors() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendAudio(audioAsset(name: "One", duration: 3))
    let second = try timeline.appendAudio(audioAsset(name: "Two", duration: 3))
    timeline.setAudioTrim(
      second.id,
      AudioTrim(startTime: 5, duration: 3, sourceOffset: 0)
    )

    timeline.setAudioStart(first.id, startTime: 1)
    #expect(abs(timeline.audioItems[0].startTime - 1) < 0.000_1)

    timeline.setAudioStart(first.id, startTime: 4)
    #expect(abs(timeline.audioItems[0].startTime - 2) < 0.000_1)
    #expect(abs(timeline.audioItems[1].startTime - 5) < 0.000_1)
  }

  @Test("Audio reorder packs the lane from its leftmost start")
  func reordersAudioItems() throws {
    var timeline = ProjectTimeline()
    let first = try timeline.appendAudio(audioAsset(name: "One", duration: 2))
    let second = try timeline.appendAudio(audioAsset(name: "Two", duration: 3))
    let third = try timeline.appendAudio(audioAsset(name: "Three", duration: 4))

    timeline.moveAudio(first.id, toIndex: 2)
    #expect(timeline.audioItems.map(\.id) == [second.id, third.id, first.id])
    #expect(abs(timeline.audioItems[0].startTime) < 0.000_1)
    #expect(abs(timeline.audioItems[1].startTime - 3) < 0.000_1)
    #expect(abs(timeline.audioItems[2].startTime - 7) < 0.000_1)
    #expect(abs(timeline.audioTrackEnd - 9) < 0.000_1)

    timeline.moveAudio(first.id, toIndex: 2)
    #expect(timeline.audioItems.map(\.id) == [second.id, third.id, first.id])
  }

  @Test("Reorder math uses midpoints of the remaining clips")
  func reorderDestinationUsesMidpoints() {
    let others: [(start: TimeInterval, duration: TimeInterval)] = [
      (0, 2),
      (2, 2),
      (4, 2),
    ]
    #expect(TimelineReorderMath.destinationIndex(dropTime: 0.4, others: others) == 0)
    #expect(TimelineReorderMath.destinationIndex(dropTime: 2.2, others: others) == 1)
    #expect(TimelineReorderMath.destinationIndex(dropTime: 5.2, others: others) == 3)
    #expect(TimelineReorderMath.destinationIndex(dropTime: 1, others: []) == 0)
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
