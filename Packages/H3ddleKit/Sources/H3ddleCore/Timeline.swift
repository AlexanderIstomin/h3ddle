import Foundation

public struct VisualItem: Identifiable, Hashable, Codable, Sendable {
  public let id: UUID
  public let assetID: AssetID
  public var duration: TimeInterval
  public var isEnabled: Bool
  public var includesNativeAudio: Bool
  /// In-point inside the source media. Images ignore this.
  public var sourceOffset: TimeInterval
  /// Empty time before this clip. Later clips stay sequential after it.
  public var gapBefore: TimeInterval

  public init(
    id: UUID = UUID(),
    assetID: AssetID,
    duration: TimeInterval,
    isEnabled: Bool = true,
    includesNativeAudio: Bool = true,
    sourceOffset: TimeInterval = 0,
    gapBefore: TimeInterval = 0
  ) {
    self.id = id
    self.assetID = assetID
    self.duration = max(0, duration)
    self.isEnabled = isEnabled
    self.includesNativeAudio = includesNativeAudio
    self.sourceOffset = max(0, sourceOffset)
    self.gapBefore = max(0, gapBefore)
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    id = try container.decode(UUID.self, forKey: .id)
    assetID = try container.decode(AssetID.self, forKey: .assetID)
    duration = max(0, try container.decode(TimeInterval.self, forKey: .duration))
    isEnabled = try container.decode(Bool.self, forKey: .isEnabled)
    includesNativeAudio = try container.decode(Bool.self, forKey: .includesNativeAudio)
    sourceOffset = max(
      0,
      try container.decodeIfPresent(TimeInterval.self, forKey: .sourceOffset) ?? 0
    )
    gapBefore = max(
      0,
      try container.decodeIfPresent(TimeInterval.self, forKey: .gapBefore) ?? 0
    )
  }

  enum CodingKeys: String, CodingKey {
    case id
    case assetID
    case duration
    case isEnabled
    case includesNativeAudio
    case sourceOffset
    case gapBefore
  }
}

public enum TimelineTrimEdge: String, Sendable {
  case leading
  case trailing
}

public typealias VisualTrimEdge = TimelineTrimEdge

public struct VisualTrim: Equatable, Sendable {
  public var duration: TimeInterval
  public var sourceOffset: TimeInterval
  public var gapBefore: TimeInterval

  public init(duration: TimeInterval, sourceOffset: TimeInterval, gapBefore: TimeInterval) {
    self.duration = max(0, duration)
    self.sourceOffset = max(0, sourceOffset)
    self.gapBefore = max(0, gapBefore)
  }
}

public enum VisualTrimMath: Sendable {
  public static func minimumDuration(framesPerSecond: Double) -> TimeInterval {
    1 / max(framesPerSecond, 1)
  }

  public static func sourceLimit(kind: MediaKind, sourceDuration: TimeInterval) -> TimeInterval? {
    kind == .video ? max(0, sourceDuration) : nil
  }

  public static func apply(
    edge: TimelineTrimEdge,
    delta: TimeInterval,
    startTime: TimeInterval,
    duration: TimeInterval,
    sourceOffset: TimeInterval,
    gapBefore: TimeInterval,
    sourceLimit: TimeInterval?,
    minimumDuration: TimeInterval
  ) -> VisualTrim {
    let duration = max(minimumDuration, duration)
    let sourceOffset = max(0, sourceOffset)
    let gapBefore = max(0, gapBefore)
    let floor = max(minimumDuration, 0.001)
    switch edge {
    case .trailing:
      var nextDuration = duration + delta
      nextDuration = max(floor, nextDuration)
      if let sourceLimit {
        nextDuration = min(nextDuration, max(floor, sourceLimit - sourceOffset))
      }
      return VisualTrim(duration: nextDuration, sourceOffset: sourceOffset, gapBefore: gapBefore)
    case .leading:
      let endTime = startTime + duration
      let previousEnd = startTime - gapBefore
      let earliestFromSource = sourceLimit == nil ? previousEnd : startTime - sourceOffset
      var nextStart = startTime + delta
      nextStart = max(max(previousEnd, earliestFromSource), nextStart)
      nextStart = min(endTime - floor, nextStart)
      return VisualTrim(
        duration: endTime - nextStart,
        sourceOffset: sourceLimit == nil ? 0 : sourceOffset + (nextStart - startTime),
        gapBefore: nextStart - previousEnd
      )
    }
  }
}

public struct AudioItem: Identifiable, Hashable, Codable, Sendable {
  public let id: UUID
  public let assetID: AssetID
  public var startTime: TimeInterval
  public var duration: TimeInterval
  public var isEnabled: Bool
  public var gain: Float
  /// In-point inside the source media.
  public var sourceOffset: TimeInterval

  public init(
    id: UUID = UUID(),
    assetID: AssetID,
    startTime: TimeInterval,
    duration: TimeInterval,
    isEnabled: Bool = true,
    gain: Float = 1,
    sourceOffset: TimeInterval = 0
  ) {
    self.id = id
    self.assetID = assetID
    self.startTime = max(0, startTime)
    self.duration = max(0, duration)
    self.isEnabled = isEnabled
    self.gain = min(max(gain, 0), 1)
    self.sourceOffset = max(0, sourceOffset)
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    id = try container.decode(UUID.self, forKey: .id)
    assetID = try container.decode(AssetID.self, forKey: .assetID)
    startTime = max(0, try container.decode(TimeInterval.self, forKey: .startTime))
    duration = max(0, try container.decode(TimeInterval.self, forKey: .duration))
    isEnabled = try container.decode(Bool.self, forKey: .isEnabled)
    gain = try container.decode(Float.self, forKey: .gain)
    sourceOffset = max(
      0,
      try container.decodeIfPresent(TimeInterval.self, forKey: .sourceOffset) ?? 0
    )
  }

  public var endTime: TimeInterval {
    startTime + duration
  }

  enum CodingKeys: String, CodingKey {
    case id
    case assetID
    case startTime
    case duration
    case isEnabled
    case gain
    case sourceOffset
  }
}

public struct AudioTrim: Equatable, Sendable {
  public var startTime: TimeInterval
  public var duration: TimeInterval
  public var sourceOffset: TimeInterval

  public init(startTime: TimeInterval, duration: TimeInterval, sourceOffset: TimeInterval) {
    self.startTime = max(0, startTime)
    self.duration = max(0, duration)
    self.sourceOffset = max(0, sourceOffset)
  }
}

public enum AudioTrimMath: Sendable {
  public static func apply(
    edge: TimelineTrimEdge,
    delta: TimeInterval,
    startTime: TimeInterval,
    duration: TimeInterval,
    sourceOffset: TimeInterval,
    sourceLimit: TimeInterval,
    earliestStart: TimeInterval,
    latestEnd: TimeInterval,
    minimumDuration: TimeInterval
  ) -> AudioTrim {
    let duration = max(minimumDuration, duration)
    let sourceOffset = max(0, sourceOffset)
    let floor = max(minimumDuration, 0.001)
    let sourceLimit = max(floor, sourceLimit)
    switch edge {
    case .trailing:
      var nextEnd = startTime + duration + delta
      nextEnd = max(startTime + floor, nextEnd)
      nextEnd = min(startTime + max(floor, sourceLimit - sourceOffset), nextEnd)
      nextEnd = min(latestEnd, nextEnd)
      return AudioTrim(
        startTime: startTime,
        duration: nextEnd - startTime,
        sourceOffset: sourceOffset
      )
    case .leading:
      let endTime = startTime + duration
      var nextStart = startTime + delta
      nextStart = max(max(0, earliestStart), nextStart)
      nextStart = max(startTime - sourceOffset, nextStart)
      nextStart = min(endTime - floor, nextStart)
      return AudioTrim(
        startTime: nextStart,
        duration: endTime - nextStart,
        sourceOffset: sourceOffset + (nextStart - startTime)
      )
    }
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
    visualItems.reduce(0) { $0 + $1.gapBefore + $1.duration }
  }

  public var visualPlacements: [VisualPlacement] {
    var cursor: TimeInterval = 0
    return visualItems.map { item in
      cursor += item.gapBefore
      let startTime = cursor
      cursor += item.duration
      return VisualPlacement(item: item, startTime: startTime)
    }
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

  public mutating func setVisualTrim(_ id: UUID, _ trim: VisualTrim) {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return }
    visualItems[index].duration = max(0, trim.duration)
    visualItems[index].sourceOffset = max(0, trim.sourceOffset)
    visualItems[index].gapBefore = max(0, trim.gapBefore)
  }

  public mutating func setAudioEnabled(_ id: UUID, isEnabled: Bool) {
    guard let index = audioItems.firstIndex(where: { $0.id == id }) else { return }
    audioItems[index].isEnabled = isEnabled
  }

  public mutating func setAudioTrim(_ id: UUID, _ trim: AudioTrim) {
    guard let index = audioItems.firstIndex(where: { $0.id == id }) else { return }
    audioItems[index].startTime = max(0, trim.startTime)
    audioItems[index].duration = max(0, trim.duration)
    audioItems[index].sourceOffset = max(0, trim.sourceOffset)
  }

  public func audioNeighborBounds(of id: UUID) -> (
    earliestStart: TimeInterval, latestEnd: TimeInterval
  ) {
    guard let item = audioItems.first(where: { $0.id == id }) else {
      return (0, .infinity)
    }
    let others = audioItems.filter { $0.id != id }
    let earliestStart =
      others
      .filter { $0.endTime <= item.startTime + 0.000_001 }
      .map(\.endTime)
      .max() ?? 0
    let latestEnd =
      others
      .filter { $0.startTime >= item.endTime - 0.000_001 }
      .map(\.startTime)
      .min() ?? .infinity
    return (earliestStart, latestEnd)
  }

  public mutating func removeVisual(_ id: UUID) {
    visualItems.removeAll { $0.id == id }
  }

  /// Removing audio deliberately preserves every later item's absolute start time.
  public mutating func removeAudio(_ id: UUID) {
    audioItems.removeAll { $0.id == id }
  }
}

public struct VisualPlacement: Hashable, Sendable {
  public var item: VisualItem
  public var startTime: TimeInterval

  public init(item: VisualItem, startTime: TimeInterval) {
    self.item = item
    self.startTime = startTime
  }
}
