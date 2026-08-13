import Foundation

public struct VisualItem: Identifiable, Hashable, Codable, Sendable {
  public let id: UUID
  public let assetID: AssetID
  public var duration: TimeInterval
  public var isEnabled: Bool
  public var includesNativeAudio: Bool

  public init(
    id: UUID = UUID(),
    assetID: AssetID,
    duration: TimeInterval,
    isEnabled: Bool = true,
    includesNativeAudio: Bool = true
  ) {
    self.id = id
    self.assetID = assetID
    self.duration = max(0, duration)
    self.isEnabled = isEnabled
    self.includesNativeAudio = includesNativeAudio
  }
}

public struct AudioItem: Identifiable, Hashable, Codable, Sendable {
  public let id: UUID
  public let assetID: AssetID
  public var startTime: TimeInterval
  public var duration: TimeInterval
  public var isEnabled: Bool
  public var gain: Float

  public init(
    id: UUID = UUID(),
    assetID: AssetID,
    startTime: TimeInterval,
    duration: TimeInterval,
    isEnabled: Bool = true,
    gain: Float = 1
  ) {
    self.id = id
    self.assetID = assetID
    self.startTime = max(0, startTime)
    self.duration = max(0, duration)
    self.isEnabled = isEnabled
    self.gain = min(max(gain, 0), 1)
  }

  public var endTime: TimeInterval {
    startTime + duration
  }
}

public enum TimelineError: Error, Equatable, Sendable {
  case expectedVisualAsset
  case expectedAudioAsset
  case missingAsset
}

public struct ProjectTimeline: Hashable, Codable, Sendable {
  public private(set) var visualItems: [VisualItem]
  public private(set) var audioItems: [AudioItem]

  public init(
    visualItems: [VisualItem] = [],
    audioItems: [AudioItem] = []
  ) {
    self.visualItems = visualItems
    self.audioItems = audioItems
  }

  public var visualDuration: TimeInterval {
    visualItems.reduce(0) { $0 + $1.duration }
  }

  public var audioTrackEnd: TimeInterval {
    audioItems.map(\.endTime).max() ?? 0
  }

  public var trailingAudioDuration: TimeInterval {
    max(0, audioTrackEnd - visualDuration)
  }

  @discardableResult
  public mutating func appendVisual(_ asset: AssetReference) throws -> VisualItem {
    guard asset.kind.isVisual else {
      throw TimelineError.expectedVisualAsset
    }
    let item = VisualItem(
      assetID: asset.id,
      duration: asset.duration,
      includesNativeAudio: asset.kind == .video
    )
    visualItems.append(item)
    return item
  }

  @discardableResult
  public mutating func appendAudio(_ asset: AssetReference) throws -> AudioItem {
    guard asset.kind == .audio else {
      throw TimelineError.expectedAudioAsset
    }
    let item = AudioItem(
      assetID: asset.id,
      startTime: audioTrackEnd,
      duration: asset.duration
    )
    audioItems.append(item)
    return item
  }

  public mutating func setVisualEnabled(_ id: UUID, isEnabled: Bool) {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return }
    visualItems[index].isEnabled = isEnabled
  }

  public mutating func setAudioEnabled(_ id: UUID, isEnabled: Bool) {
    guard let index = audioItems.firstIndex(where: { $0.id == id }) else { return }
    audioItems[index].isEnabled = isEnabled
  }

  public mutating func removeVisual(_ id: UUID) {
    visualItems.removeAll { $0.id == id }
  }

  /// Removing audio deliberately preserves every later item's absolute start time.
  public mutating func removeAudio(_ id: UUID) {
    audioItems.removeAll { $0.id == id }
  }
}
