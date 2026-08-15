import CoreGraphics
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
  public var textItems: [TextItem]
  /// Export span when the text lane is included (`max(visual, text)`).
  public var duration: TimeInterval
  public var trailingAudioDuration: TimeInterval

  public init(project: H3ddleProject) {
    visualSegments = project.timeline.visualPlacements.map { placement in
      PlannedVisualSegment(item: placement.item, startTime: placement.startTime)
    }
    audioItems = project.timeline.audioItems
    textItems = project.timeline.textItems
    duration = project.timeline.exportDuration(includeTextLane: true)
    trailingAudioDuration = max(0, project.timeline.audioTrackEnd - duration)
  }

  public func exportDuration(includeTextLane: Bool) -> TimeInterval {
    let visual = visualSegments.last.map { $0.startTime + $0.item.duration } ?? 0
    guard includeTextLane else { return visual }
    let textEnd = textItems.map(\.endTime).max() ?? 0
    return max(visual, textEnd)
  }
}

public struct ExportPreviewImage: @unchecked Sendable {
  public let image: CGImage

  public init(image: CGImage) {
    self.image = image
  }
}

public enum MediaExportEvent: Sendable {
  case preparing
  case progress(phase: String, fraction: Double)
  case preview(ExportPreviewImage)
  case completed(URL)
}

public enum MediaExportError: Error, Equatable, Sendable {
  case emptyProgram
  case cancelled
  case failed(String)
}

public protocol ProgramExporting: Sendable {
  func export(
    project: H3ddleProject,
    settings: ProgramExportSettings,
    destination: URL
  ) -> AsyncThrowingStream<MediaExportEvent, any Error>
}

extension ProgramCompositionPlan {
  public func requiresTrailingAudioWarning(range: ProgramExportRange) -> Bool {
    let end = range.resolved(in: duration).outSec
    return audioItems.contains { item in
      item.isEnabled && item.endTime > end + 0.05
    }
  }

  public func trailingAudioPast(range: ProgramExportRange) -> TimeInterval {
    let end = range.resolved(in: duration).outSec
    let audioEnd = audioItems.filter(\.isEnabled).map(\.endTime).max() ?? 0
    return max(0, audioEnd - end)
  }
}
