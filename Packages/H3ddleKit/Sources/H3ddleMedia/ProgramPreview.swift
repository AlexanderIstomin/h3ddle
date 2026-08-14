import Foundation
import H3ddleCore

public enum ProgramVisualPresentation: Equatable, Sendable {
  case video(asset: AssetReference, localTime: TimeInterval, includesNativeAudio: Bool)
  case image(asset: AssetReference)
  case empty
}

public struct ProgramAudioPresentation: Equatable, Sendable {
  public var asset: AssetReference
  public var localTime: TimeInterval

  public init(asset: AssetReference, localTime: TimeInterval) {
    self.asset = asset
    self.localTime = localTime
  }
}

public struct ProgramPreviewFrame: Equatable, Sendable {
  public var time: TimeInterval
  public var duration: TimeInterval
  public var visual: ProgramVisualPresentation
  public var visualTransform: VisualCanvasTransform
  public var audio: [ProgramAudioPresentation]

  public init(
    time: TimeInterval,
    duration: TimeInterval,
    visual: ProgramVisualPresentation,
    visualTransform: VisualCanvasTransform = .identity,
    audio: [ProgramAudioPresentation]
  ) {
    self.time = time
    self.duration = duration
    self.visual = visual
    self.visualTransform = visualTransform
    self.audio = audio
  }
}

public enum ProgramPreview {
  public static func frame(
    at time: TimeInterval,
    project: H3ddleProject,
    visualMuted: Bool = false,
    audioMuted: Bool = false
  ) -> ProgramPreviewFrame {
    let plan = ProgramCompositionPlan(project: project)
    let query = queryTime(time, duration: plan.duration)
    var visual: ProgramVisualPresentation = .empty
    var visualTransform = VisualCanvasTransform.identity

    if let segment = segment(at: query, in: plan) {
      visualTransform = segment.item.canvasTransform
      if !visualMuted, segment.item.isEnabled, let asset = project.asset(id: segment.item.assetID) {
        let localTime = max(0, query - segment.startTime) + segment.item.sourceOffset
        switch asset.kind {
        case .video:
          visual = .video(
            asset: asset,
            localTime: localTime,
            includesNativeAudio: segment.item.includesNativeAudio
          )
        case .image:
          visual = .image(asset: asset)
        case .audio:
          visual = .empty
        }
      }
    }

    let audio: [ProgramAudioPresentation]
    if audioMuted {
      audio = []
    } else {
      audio = plan.audioItems.compactMap { item in
        guard item.isEnabled, query >= item.startTime, query < item.endTime else {
          return nil
        }
        guard let asset = project.asset(id: item.assetID) else { return nil }
        return ProgramAudioPresentation(
          asset: asset,
          localTime: query - item.startTime + item.sourceOffset
        )
      }
    }

    return ProgramPreviewFrame(
      time: ProgramClock.clamp(time, duration: plan.duration),
      duration: plan.duration,
      visual: visual,
      visualTransform: visualTransform,
      audio: audio
    )
  }

  public static func visualSegmentStart(
    at time: TimeInterval,
    project: H3ddleProject
  ) -> TimeInterval? {
    let plan = ProgramCompositionPlan(project: project)
    return segment(at: queryTime(time, duration: plan.duration), in: plan)?.startTime
  }

  private static func queryTime(_ time: TimeInterval, duration: TimeInterval) -> TimeInterval {
    guard duration > 0 else { return 0 }
    return min(max(0, time), duration - 0.000_001)
  }

  private static func segment(
    at time: TimeInterval,
    in plan: ProgramCompositionPlan
  ) -> PlannedVisualSegment? {
    plan.visualSegments.first { segment in
      time >= segment.startTime && time < segment.startTime + segment.item.duration
    }
  }
}
