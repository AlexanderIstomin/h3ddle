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

public struct ProgramVisualMix: Equatable, Sendable {
  public var outgoing: ProgramVisualPresentation
  public var outgoingTransform: VisualCanvasTransform
  public var outgoingEffects: [VisualEffectInstance]
  public var incoming: ProgramVisualPresentation
  public var incomingTransform: VisualCanvasTransform
  public var incomingEffects: [VisualEffectInstance]
  public var progress: Double
  public var kind: VisualTransitionKind

  public init(
    outgoing: ProgramVisualPresentation,
    outgoingTransform: VisualCanvasTransform,
    outgoingEffects: [VisualEffectInstance] = [],
    incoming: ProgramVisualPresentation,
    incomingTransform: VisualCanvasTransform,
    incomingEffects: [VisualEffectInstance] = [],
    progress: Double,
    kind: VisualTransitionKind
  ) {
    self.outgoing = outgoing
    self.outgoingTransform = outgoingTransform
    self.outgoingEffects = outgoingEffects
    self.incoming = incoming
    self.incomingTransform = incomingTransform
    self.incomingEffects = incomingEffects
    self.progress = min(max(progress, 0), 1)
    self.kind = kind
  }
}

public struct ProgramPreviewFrame: Equatable, Sendable {
  public var time: TimeInterval
  public var duration: TimeInterval
  public var visual: ProgramVisualPresentation
  public var visualTransform: VisualCanvasTransform
  public var visualEffects: [VisualEffectInstance]
  public var transition: ProgramVisualMix?
  public var audio: [ProgramAudioPresentation]

  public init(
    time: TimeInterval,
    duration: TimeInterval,
    visual: ProgramVisualPresentation,
    visualTransform: VisualCanvasTransform = .identity,
    visualEffects: [VisualEffectInstance] = [],
    transition: ProgramVisualMix? = nil,
    audio: [ProgramAudioPresentation]
  ) {
    self.time = time
    self.duration = duration
    self.visual = visual
    self.visualTransform = visualTransform
    self.visualEffects = visualEffects
    self.transition = transition
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
    var visualEffects: [VisualEffectInstance] = []
    var transition: ProgramVisualMix?

    if let segment = segment(at: query, in: plan) {
      visualTransform = segment.item.canvasTransform
      visualEffects = segment.item.effects
      if !visualMuted, segment.item.isEnabled, let asset = project.asset(id: segment.item.assetID) {
        let localTime = max(0, query - segment.startTime) + segment.item.sourceOffset
        visual = presentation(asset: asset, item: segment.item, localTime: localTime)
      }
      if !visualMuted {
        transition = mix(at: query, incoming: segment, project: project, plan: plan)
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
      visualEffects: visualEffects,
      transition: transition,
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
    // Overlapping transitions match two segments; prefer the incoming clip so
    // the mix can read its authored transition.
    plan.visualSegments.last { segment in
      time >= segment.startTime && time < segment.startTime + segment.item.duration
    }
  }

  private static func mix(
    at time: TimeInterval,
    incoming: PlannedVisualSegment,
    project: H3ddleProject,
    plan: ProgramCompositionPlan
  ) -> ProgramVisualMix? {
    guard incoming.item.isEnabled, incoming.item.gapBefore == 0,
      let authored = incoming.item.transition
    else {
      return nil
    }
    guard let outgoingIndex = plan.visualSegments.firstIndex(where: { $0.item.id == incoming.item.id }),
      outgoingIndex > 0
    else {
      return nil
    }
    let outgoing = plan.visualSegments[outgoingIndex - 1]
    guard outgoing.item.isEnabled else { return nil }
    let duration = VisualTransitionMath.resolvedDuration(
      authored.duration,
      outgoing: outgoing.item.duration,
      incoming: incoming.item.duration
    )
    guard
      let progress = VisualTransitionMath.progress(
        at: time,
        cut: incoming.startTime,
        duration: duration
      )
    else {
      return nil
    }
    guard let outgoingAsset = project.asset(id: outgoing.item.assetID),
      let incomingAsset = project.asset(id: incoming.item.assetID)
    else {
      return nil
    }
    let elapsed = time - incoming.startTime
    let outgoingLocal = outgoing.item.sourceOffset + (outgoing.item.duration - duration) + elapsed
    let incomingLocal = incoming.item.sourceOffset + elapsed
    return ProgramVisualMix(
      outgoing: presentation(asset: outgoingAsset, item: outgoing.item, localTime: outgoingLocal),
      outgoingTransform: outgoing.item.canvasTransform,
      outgoingEffects: outgoing.item.effects,
      incoming: presentation(asset: incomingAsset, item: incoming.item, localTime: incomingLocal),
      incomingTransform: incoming.item.canvasTransform,
      incomingEffects: incoming.item.effects,
      progress: progress,
      kind: authored.kind
    )
  }

  private static func presentation(
    asset: AssetReference,
    item: VisualItem,
    localTime: TimeInterval
  ) -> ProgramVisualPresentation {
    switch asset.kind {
    case .video:
      return .video(
        asset: asset,
        localTime: max(0, localTime),
        includesNativeAudio: item.includesNativeAudio
      )
    case .image:
      return .image(asset: asset)
    case .audio:
      return .empty
    }
  }
}
