import Foundation
import H3ddleCore
import Testing

@testable import H3ddleMedia

@Suite("Program preview")
struct ProgramPreviewTests {
  @Test("Enabled video occupies derived program time")
  func mapsEnabledVideo() throws {
    var project = H3ddleProject()
    let first = videoAsset(name: "One", duration: 5)
    let second = videoAsset(name: "Two", duration: 4)
    project.addAsset(first)
    project.addAsset(second)
    try project.timeline.appendVisual(first)
    try project.timeline.appendVisual(second)

    let frame = ProgramPreview.frame(at: 6, project: project)
    guard case .video(let asset, let localTime, let includesNativeAudio) = frame.visual else {
      Issue.record("Expected a video presentation")
      return
    }
    #expect(asset.id == second.id)
    #expect(abs(localTime - 1) < 0.000_1)
    #expect(includesNativeAudio)
    #expect(frame.duration == 9)
    #expect(frame.visualTransform == .identity)
  }

  @Test("Preview carries the current visual canvas transform")
  func carriesVisualCanvasTransform() throws {
    var project = H3ddleProject()
    let first = videoAsset(name: "One", duration: 5)
    let second = videoAsset(name: "Two", duration: 4)
    project.addAsset(first)
    project.addAsset(second)
    let firstItem = try project.timeline.appendVisual(first)
    let secondItem = try project.timeline.appendVisual(second)
    project.timeline.setVisualCanvasFit(firstItem.id, .cover)
    project.timeline.rotateVisual(firstItem.id)
    project.timeline.rotateVisual(firstItem.id)
    project.timeline.setVisualCanvasFit(secondItem.id, .cover)

    let firstFrame = ProgramPreview.frame(at: 2, project: project)
    #expect(firstFrame.visualTransform.fit == .cover)
    #expect(firstFrame.visualTransform.rotationTurns == 2)
    #expect(abs(firstFrame.visualTransform.rotationRadians - .pi) < 0.000_1)

    let secondFrame = ProgramPreview.frame(at: 6, project: project)
    #expect(secondFrame.visualTransform.fit == .cover)
    #expect(secondFrame.visualTransform.rotationTurns == 0)
  }

  @Test("Transform overrides replace the committed visual transform")
  func transformOverridesReplaceCommitted() throws {
    var project = H3ddleProject()
    let video = videoAsset(name: "One", duration: 4)
    project.addAsset(video)
    let item = try project.timeline.appendVisual(video)
    project.timeline.setVisualCanvasFit(item.id, .cover)

    let override = CanvasObjectTransform(
      fit: .fit,
      translationX: 0.1,
      translationY: -0.2,
      scale: 1.5,
      rotationRadians: 0.4
    )
    let frame = ProgramPreview.frame(
      at: 1,
      project: project,
      transformOverrides: [item.id: override]
    )
    #expect(frame.visualTransform == override)
    #expect(abs(frame.previewDuration - 4) < 0.000_1)
    #expect(abs(frame.duration - 4) < 0.000_1)
  }

  @Test("Preview query can sit past the last visual when audio outlasts it")
  func previewQueryPastVisualIsEmpty() throws {
    var project = H3ddleProject()
    let visual = videoAsset(name: "Picture", duration: 2)
    let audio = audioAsset(name: "Score", duration: 5)
    project.addAsset(visual)
    project.addAsset(audio)
    try project.timeline.appendVisual(visual)
    try project.timeline.appendAudio(audio)

    let frame = ProgramPreview.frame(at: 3, project: project)
    #expect(frame.visual == .empty)
    #expect(abs(frame.duration - 2) < 0.000_1)
    #expect(abs(frame.previewDuration - 5) < 0.000_1)
    #expect(frame.audio.map(\.asset.id) == [audio.id])
    #expect(ProgramPreview.visualSegmentStart(at: 3, project: project) == nil)
  }

  @Test("An incoming dissolve reports both sources and progress")
  func dissolveReportsBothSources() throws {
    var project = H3ddleProject()
    let first = videoAsset(name: "A", duration: 4)
    let second = videoAsset(name: "B", duration: 4)
    project.addAsset(first)
    project.addAsset(second)
    try project.timeline.appendVisual(first)
    let incoming = try project.timeline.appendVisual(second)
    project.timeline.setVisualTransition(
      incoming.id,
      VisualTransition(kind: .dissolve, duration: 1)
    )

    #expect(ProgramPreview.frame(at: 2.9, project: project).transition == nil)
    let mix = ProgramPreview.frame(at: 3.25, project: project).transition
    #expect(mix?.kind == .dissolve)
    #expect(abs((mix?.progress ?? -1) - 0.25) < 0.000_1)
    guard case .video(let outgoing, let outgoingTime, _) = mix?.outgoing else {
      Issue.record("Expected the outgoing video")
      return
    }
    guard case .video(let incomingAsset, let incomingTime, _) = mix?.incoming else {
      Issue.record("Expected the incoming video")
      return
    }
    #expect(outgoing.id == first.id)
    #expect(incomingAsset.id == second.id)
    #expect(abs(outgoingTime - 3.25) < 0.000_1)
    #expect(abs(incomingTime - 0.25) < 0.000_1)
    #expect(ProgramPreview.frame(at: 4.1, project: project).transition == nil)
    #expect(abs(ProgramPreview.frame(at: 3.25, project: project).duration - 7) < 0.000_1)
  }

  @Test("Disabled visuals keep duration and render empty")
  func disabledVisualKeepsDuration() throws {
    var project = H3ddleProject()
    let first = videoAsset(name: "One", duration: 5)
    let second = videoAsset(name: "Two", duration: 4)
    project.addAsset(first)
    project.addAsset(second)
    let firstItem = try project.timeline.appendVisual(first)
    try project.timeline.appendVisual(second)
    project.timeline.setVisualEnabled(firstItem.id, isEnabled: false)

    let duringDisabled = ProgramPreview.frame(at: 2, project: project)
    #expect(duringDisabled.visual == .empty)
    #expect(duringDisabled.duration == 9)

    let duringSecond = ProgramPreview.frame(at: 6, project: project)
    guard case .video(let asset, _, _) = duringSecond.visual else {
      Issue.record("Expected the second visual after the disabled hole")
      return
    }
    #expect(asset.id == second.id)
  }

  @Test("Audio is resolved at absolute start times including gaps")
  func audioUsesAbsoluteTime() throws {
    var project = H3ddleProject()
    let visual = videoAsset(name: "Picture", duration: 10)
    let first = audioAsset(name: "One", duration: 3)
    let second = audioAsset(name: "Two", duration: 2)
    project.addAsset(visual)
    project.addAsset(first)
    project.addAsset(second)
    try project.timeline.appendVisual(visual)
    let firstItem = try project.timeline.appendAudio(first)
    try project.timeline.appendAudio(second)
    project.timeline.removeAudio(firstItem.id)

    #expect(ProgramPreview.frame(at: 1, project: project).audio.isEmpty)
    let later = ProgramPreview.frame(at: 3.5, project: project)
    #expect(later.audio.map(\.asset.id) == [second.id])
    #expect(abs((later.audio.first?.localTime ?? -1) - 0.5) < 0.000_1)
  }

  @Test("Audio source offset shifts the local audio time")
  func audioSourceOffsetShiftsLocalTime() throws {
    var project = H3ddleProject()
    let visual = videoAsset(name: "Picture", duration: 10)
    let audio = audioAsset(name: "Score", duration: 6)
    project.addAsset(visual)
    project.addAsset(audio)
    try project.timeline.appendVisual(visual)
    let item = try project.timeline.appendAudio(audio)
    project.timeline.setAudioTrim(
      item.id,
      AudioTrim(startTime: 2, duration: 3, sourceOffset: 1.25)
    )

    let frame = ProgramPreview.frame(at: 3, project: project)
    #expect(frame.audio.map(\.asset.id) == [audio.id])
    #expect(abs((frame.audio.first?.localTime ?? -1) - 2.25) < 0.000_1)
  }

  @Test("Visual source offset shifts the local video time")
  func visualSourceOffsetShiftsLocalTime() throws {
    var project = H3ddleProject()
    let video = videoAsset(name: "Offset", duration: 6)
    project.addAsset(video)
    let item = try project.timeline.appendVisual(video)
    project.timeline.setVisualTrim(
      item.id,
      VisualTrim(duration: 4, sourceOffset: 1.5, gapBefore: 1)
    )

    #expect(ProgramPreview.frame(at: 0.4, project: project).visual == .empty)
    let frame = ProgramPreview.frame(at: 2, project: project)
    guard case .video(let asset, let localTime, _) = frame.visual else {
      Issue.record("Expected a trimmed video presentation")
      return
    }
    #expect(asset.id == video.id)
    #expect(abs(localTime - 2.5) < 0.000_1)
    #expect(abs(frame.duration - 5) < 0.000_1)
  }

  @Test("Muted lanes produce silence or a black frame")
  func mutedLanesSuppressMedia() throws {
    var project = H3ddleProject()
    let visual = videoAsset(name: "Picture", duration: 4)
    let audio = audioAsset(name: "Score", duration: 4)
    project.addAsset(visual)
    project.addAsset(audio)
    try project.timeline.appendVisual(visual)
    try project.timeline.appendAudio(audio)

    let mutedVisual = ProgramPreview.frame(at: 1, project: project, visualMuted: true)
    #expect(mutedVisual.visual == .empty)
    #expect(mutedVisual.audio.map(\.asset.id) == [audio.id])

    let mutedAudio = ProgramPreview.frame(at: 1, project: project, audioMuted: true)
    #expect(mutedAudio.audio.isEmpty)
    guard case .video = mutedAudio.visual else {
      Issue.record("Expected the visual to remain when only audio is muted")
      return
    }
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
