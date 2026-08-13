import Foundation
import H3ddleCore

public struct PlannedVisualSegment: Hashable, Sendable {
  public var item: VisualItem
  public var startTime: TimeInterval

  public init(item: VisualItem, startTime: TimeInterval) {
    self.item = item
    self.startTime = startTime
  }
}

public struct ProgramCompositionPlan: Hashable, Sendable {
  public var visualSegments: [PlannedVisualSegment]
  public var audioItems: [AudioItem]
  public var duration: TimeInterval
  public var trailingAudioDuration: TimeInterval

  public init(project: H3ddleProject) {
    var cursor: TimeInterval = 0
    visualSegments = project.timeline.visualItems.map { item in
      defer { cursor += item.duration }
      return PlannedVisualSegment(item: item, startTime: cursor)
    }
    audioItems = project.timeline.audioItems
    duration = project.timeline.visualDuration
    trailingAudioDuration = project.timeline.trailingAudioDuration
  }
}

public enum MediaExportEvent: Hashable, Sendable {
  case preparing
  case progress(Double)
  case completed(URL)
}

public protocol ProgramExporting: Sendable {
  func export(
    project: H3ddleProject,
    destination: URL
  ) -> AsyncThrowingStream<MediaExportEvent, any Error>
}
