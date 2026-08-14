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
  public var duration: TimeInterval
  public var trailingAudioDuration: TimeInterval

  public init(project: H3ddleProject) {
    visualSegments = project.timeline.visualPlacements.map { placement in
      PlannedVisualSegment(item: placement.item, startTime: placement.startTime)
    }
    audioItems = project.timeline.audioItems
    duration = project.timeline.visualDuration
    trailingAudioDuration = project.timeline.trailingAudioDuration
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
