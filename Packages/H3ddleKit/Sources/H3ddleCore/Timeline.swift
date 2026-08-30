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
  /// How this clip fills the program canvas.
  public var canvasFit: CanvasFit
  /// Nearest quarter-turn hint. `rotationRadians` is the source of truth.
  public var rotationTurns: Int
  /// Normalized translation. +x is right, +y is up.
  public var translationX: Double
  public var translationY: Double
  public var uniformScale: Double
  public var rotationRadians: Double
  /// Mix from the previous adjacent visual into this clip.
  public var transition: VisualTransition?
  /// Ordered filter stack drawn after canvas placement.
  public var effects: [VisualEffectInstance]

  public var canvasTransform: CanvasObjectTransform {
    get {
      CanvasObjectTransform(
        fit: canvasFit,
        translationX: translationX,
        translationY: translationY,
        scale: uniformScale,
        rotationRadians: rotationRadians
      )
    }
    set {
      canvasFit = newValue.fit
      translationX = newValue.translationX
      translationY = newValue.translationY
      uniformScale = newValue.scale
      rotationRadians = newValue.rotationRadians
      rotationTurns = newValue.rotationTurns
    }
  }

  public init(
    id: UUID = UUID(),
    assetID: AssetID,
    duration: TimeInterval,
    isEnabled: Bool = true,
    includesNativeAudio: Bool = true,
    sourceOffset: TimeInterval = 0,
    gapBefore: TimeInterval = 0,
    canvasFit: CanvasFit = .fit,
    rotationTurns: Int = 0,
    translationX: Double = 0,
    translationY: Double = 0,
    uniformScale: Double = 1,
    rotationRadians: Double? = nil,
    transition: VisualTransition? = nil,
    effects: [VisualEffectInstance] = []
  ) {
    self.id = id
    self.assetID = assetID
    self.duration = max(0, duration)
    self.isEnabled = isEnabled
    self.includesNativeAudio = includesNativeAudio
    self.sourceOffset = max(0, sourceOffset)
    self.gapBefore = max(0, gapBefore)
    self.canvasFit = canvasFit
    self.translationX = translationX
    self.translationY = translationY
    self.uniformScale = max(uniformScale, 0.01)
    if let rotationRadians {
      self.rotationRadians = rotationRadians
      self.rotationTurns = CanvasObjectTransform(rotationRadians: rotationRadians).rotationTurns
    } else {
      let turns = CanvasLayout.normalizedTurns(rotationTurns)
      self.rotationTurns = turns
      self.rotationRadians = Double(turns) * .pi / 2
    }
    self.transition = transition
    self.effects = effects
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
    canvasFit = try container.decodeIfPresent(CanvasFit.self, forKey: .canvasFit) ?? .fit
    let decodedTurns = CanvasLayout.normalizedTurns(
      try container.decodeIfPresent(Int.self, forKey: .rotationTurns) ?? 0
    )
    translationX = try container.decodeIfPresent(Double.self, forKey: .translationX) ?? 0
    translationY = try container.decodeIfPresent(Double.self, forKey: .translationY) ?? 0
    uniformScale = max(
      try container.decodeIfPresent(Double.self, forKey: .uniformScale) ?? 1,
      0.01
    )
    if let radians = try container.decodeIfPresent(Double.self, forKey: .rotationRadians) {
      rotationRadians = radians
      rotationTurns = CanvasObjectTransform(rotationRadians: radians).rotationTurns
    } else {
      rotationTurns = decodedTurns
      rotationRadians = Double(decodedTurns) * .pi / 2
    }
    transition = try container.decodeIfPresent(VisualTransition.self, forKey: .transition)
    effects = try container.decodeIfPresent([VisualEffectInstance].self, forKey: .effects) ?? []
  }

  enum CodingKeys: String, CodingKey {
    case id
    case assetID
    case duration
    case isEnabled
    case includesNativeAudio
    case sourceOffset
    case gapBefore
    case canvasFit
    case rotationTurns
    case translationX
    case translationY
    case uniformScale
    case rotationRadians
    case transition
    case effects
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

public struct TextTrim: Equatable, Sendable {
  public var startTime: TimeInterval
  public var duration: TimeInterval

  public init(startTime: TimeInterval, duration: TimeInterval) {
    self.startTime = max(0, startTime)
    self.duration = max(0, duration)
  }
}

public enum TextTrimMath: Sendable {
  public static func apply(
    edge: TimelineTrimEdge,
    delta: TimeInterval,
    startTime: TimeInterval,
    duration: TimeInterval,
    minimumDuration: TimeInterval
  ) -> TextTrim {
    let floor = max(minimumDuration, 0.001)
    switch edge {
    case .trailing:
      return TextTrim(startTime: startTime, duration: max(floor, duration + delta))
    case .leading:
      let endTime = startTime + duration
      var nextStart = max(0, startTime + delta)
      nextStart = min(endTime - floor, nextStart)
      return TextTrim(startTime: nextStart, duration: endTime - nextStart)
    }
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
  public private(set) var textItems: [TextItem]

  public init(
    visualItems: [VisualItem] = [],
    audioItems: [AudioItem] = [],
    textItems: [TextItem] = []
  ) {
    self.visualItems = visualItems
    self.audioItems = audioItems
    self.textItems = textItems
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    visualItems = try container.decodeIfPresent([VisualItem].self, forKey: .visualItems) ?? []
    audioItems = try container.decodeIfPresent([AudioItem].self, forKey: .audioItems) ?? []
    textItems = try container.decodeIfPresent([TextItem].self, forKey: .textItems) ?? []
  }

  enum CodingKeys: String, CodingKey {
    case visualItems
    case audioItems
    case textItems
  }

  public var visualDuration: TimeInterval {
    visualPlacements.last.map { $0.startTime + $0.item.duration } ?? 0
  }

  public var visualPlacements: [VisualPlacement] {
    var cursor: TimeInterval = 0
    return visualItems.enumerated().map { index, item in
      cursor += item.gapBefore
      cursor = max(0, cursor - transitionOverlap(at: index))
      let startTime = cursor
      cursor += item.duration
      return VisualPlacement(item: item, startTime: startTime)
    }
  }

  /// How far the incoming clip pulls into the previous clip. Zero if the cut
  /// cannot hold a transition.
  public func transitionOverlap(of id: UUID) -> TimeInterval {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return 0 }
    return transitionOverlap(at: index)
  }

  private func transitionOverlap(at index: Int) -> TimeInterval {
    guard index > 0 else { return 0 }
    let item = visualItems[index]
    guard item.gapBefore == 0, let transition = item.transition else { return 0 }
    return VisualTransitionMath.resolvedDuration(
      transition.duration,
      outgoing: visualItems[index - 1].duration,
      incoming: item.duration
    )
  }

  public var audioTrackEnd: TimeInterval {
    audioItems.map(\.endTime).max() ?? 0
  }

  public var trailingAudioDuration: TimeInterval {
    max(0, audioTrackEnd - visualDuration)
  }

  public var textTrackEnd: TimeInterval {
    textItems.map(\.endTime).max() ?? 0
  }

  public var trailingTextDuration: TimeInterval {
    max(0, textTrackEnd - visualDuration)
  }

  public func exportDuration(includeTextLane: Bool) -> TimeInterval {
    guard includeTextLane else { return visualDuration }
    return max(visualDuration, textTrackEnd)
  }

  @discardableResult
  public mutating func appendVisual(
    _ asset: AssetReference,
    includesNativeAudio: Bool? = nil
  ) throws -> VisualItem {
    guard asset.kind.isVisual else {
      throw TimelineError.expectedVisualAsset
    }
    let item = VisualItem(
      assetID: asset.id,
      duration: asset.duration,
      includesNativeAudio: includesNativeAudio ?? (asset.kind == .video)
    )
    visualItems.append(item)
    return item
  }

  @discardableResult
  public mutating func appendAudio(_ asset: AssetReference) throws -> AudioItem {
    try placeAudio(asset, at: audioTrackEnd)
  }

  @discardableResult
  public mutating func insertVisual(
    _ asset: AssetReference,
    at index: Int,
    includesNativeAudio: Bool? = nil
  ) throws -> VisualItem {
    guard asset.kind.isVisual else {
      throw TimelineError.expectedVisualAsset
    }
    let item = VisualItem(
      assetID: asset.id,
      duration: asset.duration,
      includesNativeAudio: includesNativeAudio ?? (asset.kind == .video)
    )
    let dest = min(max(0, index), visualItems.count)
    visualItems.insert(item, at: dest)
    sanitizeVisualTransitions()
    return item
  }

  @discardableResult
  public mutating func placeAudio(
    _ asset: AssetReference,
    at startTime: TimeInterval
  ) throws -> AudioItem {
    guard asset.kind == .audio else {
      throw TimelineError.expectedAudioAsset
    }
    let item = AudioItem(
      assetID: asset.id,
      startTime: max(0, startTime),
      duration: asset.duration
    )
    audioItems.append(item)
    setAudioStart(item.id, startTime: startTime)
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

  public mutating func setVisualCanvasFit(_ id: UUID, _ fit: CanvasFit) {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return }
    var transform = visualItems[index].canvasTransform
    transform.fit = fit
    transform.translationX = 0
    transform.translationY = 0
    transform.scale = 1
    visualItems[index].canvasTransform = transform
  }

  public mutating func rotateVisual(_ id: UUID, by turns: Int = 1) {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return }
    visualItems[index].canvasTransform = visualItems[index].canvasTransform.rotated(by: turns)
  }

  public mutating func setVisualCanvasTransform(_ id: UUID, _ transform: CanvasObjectTransform) {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return }
    visualItems[index].canvasTransform = transform
  }

  public mutating func resetVisualTransform(_ id: UUID) {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return }
    var transform = visualItems[index].canvasTransform
    transform.translationX = 0
    transform.translationY = 0
    transform.scale = 1
    visualItems[index].canvasTransform = transform
  }

  public mutating func setVisualIncludesNativeAudio(_ id: UUID, includes: Bool) {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return }
    visualItems[index].includesNativeAudio = includes
  }

  public func canApplyVisualTransition(_ id: UUID) -> Bool {
    guard let index = visualItems.firstIndex(where: { $0.id == id }), index > 0 else {
      return false
    }
    let incoming = visualItems[index]
    let outgoing = visualItems[index - 1]
    return incoming.gapBefore == 0 && outgoing.duration > 0 && incoming.duration > 0
  }

  public func maximumVisualTransitionDuration(of id: UUID) -> TimeInterval {
    guard let index = visualItems.firstIndex(where: { $0.id == id }), index > 0 else {
      return 0
    }
    return VisualTransitionMath.maximumDuration(
      outgoing: visualItems[index - 1].duration,
      incoming: visualItems[index].duration
    )
  }

  public mutating func setVisualTransition(_ id: UUID, _ transition: VisualTransition?) {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return }
    guard let transition else {
      visualItems[index].transition = nil
      return
    }
    guard canApplyVisualTransition(id) else { return }
    let duration = VisualTransitionMath.resolvedDuration(
      transition.duration,
      outgoing: visualItems[index - 1].duration,
      incoming: visualItems[index].duration
    )
    guard duration > 0.000_1 else {
      visualItems[index].transition = nil
      return
    }
    visualItems[index].transition = VisualTransition(kind: transition.kind, duration: duration)
  }

  @discardableResult
  public mutating func addVisualEffect(_ id: UUID, kind: VisualEffectKind) -> VisualEffectInstance? {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return nil }
    let effect = VisualEffectInstance(kind: kind)
    visualItems[index].effects.append(effect)
    return effect
  }

  public mutating func setVisualEffect(_ id: UUID, effect: VisualEffectInstance) {
    guard let clip = visualItems.firstIndex(where: { $0.id == id }),
      let index = visualItems[clip].effects.firstIndex(where: { $0.id == effect.id })
    else {
      return
    }
    visualItems[clip].effects[index] = effect
  }

  public mutating func removeVisualEffect(_ id: UUID, effectID: UUID) {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return }
    visualItems[index].effects.removeAll { $0.id == effectID }
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

  /// Slides a clip without overlapping neighbors. Later clips stay put.
  public mutating func setAudioStart(_ id: UUID, startTime: TimeInterval) {
    guard let index = audioItems.firstIndex(where: { $0.id == id }) else { return }
    let bounds = audioNeighborBounds(of: id)
    let duration = audioItems[index].duration
    var start = max(0, max(bounds.earliestStart, startTime))
    if bounds.latestEnd.isFinite {
      start = min(start, max(bounds.earliestStart, bounds.latestEnd - duration))
    }
    audioItems[index].startTime = start
  }

  /// Inserts a copy at the source's out-point. Later clips that would overlap
  /// are pushed by the copy's duration; a large enough gap is left alone.
  @discardableResult
  public mutating func duplicateAudio(_ id: UUID) -> AudioItem? {
    guard let index = audioItems.firstIndex(where: { $0.id == id }) else { return nil }
    let item = audioItems[index]
    let start = item.endTime
    let copyEnd = start + item.duration
    let overlaps = audioItems.contains { other in
      other.id != id
        && other.startTime < copyEnd - 0.000_001
        && other.endTime > start + 0.000_001
    }
    if overlaps {
      for i in audioItems.indices
      where audioItems[i].id != id && audioItems[i].startTime >= start - 0.000_001 {
        audioItems[i].startTime += item.duration
      }
    }
    let copy = AudioItem(
      assetID: item.assetID,
      startTime: start,
      duration: item.duration,
      isEnabled: item.isEnabled,
      gain: item.gain,
      sourceOffset: item.sourceOffset
    )
    audioItems.insert(copy, at: index + 1)
    return copy
  }

  /// Reorders the audio lane by start time and packs the sequence from its
  /// previous leftmost start so clips stay contiguous and non-overlapping.
  public mutating func moveAudio(_ id: UUID, toIndex: Int) {
    var sorted = audioItems.sorted(by: TimelineReorderMath.audioOrder)
    guard let from = sorted.firstIndex(where: { $0.id == id }) else { return }
    let dest = min(max(0, toIndex), sorted.count - 1)
    if dest == from { return }
    let sequenceStart = sorted.map(\.startTime).min() ?? 0
    let item = sorted.remove(at: from)
    sorted.insert(item, at: dest)
    var cursor = sequenceStart
    for i in sorted.indices {
      sorted[i].startTime = cursor
      cursor = sorted[i].endTime
    }
    audioItems = sorted
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

  /// Inserts a copy immediately after the original. The copy keeps trim,
  /// canvas transform, native-audio, and a new effect-id stack. It does not
  /// inherit the original's incoming transition.
  @discardableResult
  public mutating func duplicateVisual(_ id: UUID) -> VisualItem? {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return nil }
    let item = visualItems[index]
    let copy = VisualItem(
      assetID: item.assetID,
      duration: item.duration,
      isEnabled: item.isEnabled,
      includesNativeAudio: item.includesNativeAudio,
      sourceOffset: item.sourceOffset,
      gapBefore: 0,
      canvasFit: item.canvasFit,
      rotationTurns: item.rotationTurns,
      translationX: item.translationX,
      translationY: item.translationY,
      uniformScale: item.uniformScale,
      rotationRadians: item.rotationRadians,
      transition: nil,
      effects: item.effects.map { $0.copying() }
    )
    visualItems.insert(copy, at: index + 1)
    return copy
  }

  /// Moves the clip so its index in the ordered list becomes `toIndex`.
  /// Incoming transitions on the moved clip and its old/new neighbors are dropped.
  public mutating func moveVisual(_ id: UUID, toIndex: Int) {
    guard let from = visualItems.firstIndex(where: { $0.id == id }) else { return }
    let dest = min(max(0, toIndex), visualItems.count - 1)
    if dest == from { return }
    if from + 1 < visualItems.count {
      visualItems[from + 1].transition = nil
    }
    var item = visualItems.remove(at: from)
    item.transition = nil
    visualItems.insert(item, at: dest)
    if dest + 1 < visualItems.count {
      visualItems[dest + 1].transition = nil
    }
    sanitizeVisualTransitions()
  }

  public mutating func removeVisual(_ id: UUID) {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return }
    if index + 1 < visualItems.count {
      visualItems[index + 1].transition = nil
    }
    visualItems.remove(at: index)
    sanitizeVisualTransitions()
  }

  private mutating func sanitizeVisualTransitions() {
    guard !visualItems.isEmpty else { return }
    visualItems[0].transition = nil
    for index in visualItems.indices {
      guard let transition = visualItems[index].transition else { continue }
      let id = visualItems[index].id
      if !canApplyVisualTransition(id) {
        visualItems[index].transition = nil
        continue
      }
      let duration = VisualTransitionMath.resolvedDuration(
        transition.duration,
        outgoing: visualItems[index - 1].duration,
        incoming: visualItems[index].duration
      )
      visualItems[index].transition =
        duration > 0.000_1
        ? VisualTransition(kind: transition.kind, duration: duration)
        : nil
    }
  }

  /// Removing audio deliberately preserves every later item's absolute start time.
  public mutating func removeAudio(_ id: UUID) {
    audioItems.removeAll { $0.id == id }
  }

  public func canSplitVisual(
    _ id: UUID,
    at time: TimeInterval,
    framesPerSecond: Double
  ) -> Bool {
    guard let placement = visualPlacements.first(where: { $0.item.id == id }) else {
      return false
    }
    return TimelineSplit.canSplit(
      startTime: placement.startTime,
      duration: placement.item.duration,
      at: time,
      framesPerSecond: framesPerSecond
    )
  }

  public func canSplitAudio(
    _ id: UUID,
    at time: TimeInterval,
    framesPerSecond: Double
  ) -> Bool {
    guard let item = audioItems.first(where: { $0.id == id }) else { return false }
    return TimelineSplit.canSplit(
      startTime: item.startTime,
      duration: item.duration,
      at: time,
      framesPerSecond: framesPerSecond
    )
  }

  /// Divides the visual clip at `time`. The left half keeps the original id.
  @discardableResult
  public mutating func splitVisual(
    _ id: UUID,
    at time: TimeInterval,
    sourceKind: MediaKind,
    framesPerSecond: Double
  ) -> (left: UUID, right: UUID)? {
    guard let index = visualItems.firstIndex(where: { $0.id == id }) else { return nil }
    let placement = visualPlacements[index]
    let item = visualItems[index]
    guard
      TimelineSplit.canSplit(
        startTime: placement.startTime,
        duration: item.duration,
        at: time,
        framesPerSecond: framesPerSecond
      )
    else {
      return nil
    }
    let leftDuration = time - placement.startTime
    let rightDuration = item.duration - leftDuration
    visualItems[index].duration = leftDuration
    let right = VisualItem(
      assetID: item.assetID,
      duration: rightDuration,
      isEnabled: item.isEnabled,
      includesNativeAudio: item.includesNativeAudio,
      sourceOffset: sourceKind == .video ? item.sourceOffset + leftDuration : 0,
      gapBefore: 0,
      canvasFit: item.canvasFit,
      rotationTurns: item.rotationTurns,
      translationX: item.translationX,
      translationY: item.translationY,
      uniformScale: item.uniformScale,
      rotationRadians: item.rotationRadians,
      transition: nil,
      effects: item.effects.map { $0.copying() }
    )
    visualItems.insert(right, at: index + 1)
    return (item.id, right.id)
  }

  /// Divides the audio clip at `time`. Later audio start times stay put.
  @discardableResult
  public mutating func splitAudio(
    _ id: UUID,
    at time: TimeInterval,
    framesPerSecond: Double
  ) -> (left: UUID, right: UUID)? {
    guard let index = audioItems.firstIndex(where: { $0.id == id }) else { return nil }
    let item = audioItems[index]
    guard
      TimelineSplit.canSplit(
        startTime: item.startTime,
        duration: item.duration,
        at: time,
        framesPerSecond: framesPerSecond
      )
    else {
      return nil
    }
    let leftDuration = time - item.startTime
    audioItems[index].duration = leftDuration
    let right = AudioItem(
      assetID: item.assetID,
      startTime: time,
      duration: item.duration - leftDuration,
      isEnabled: item.isEnabled,
      gain: item.gain,
      sourceOffset: item.sourceOffset + leftDuration
    )
    audioItems.insert(right, at: index + 1)
    return (item.id, right.id)
  }

  @discardableResult
  public mutating func insertText(_ item: TextItem) -> TextItem {
    var item = item
    item.startTime = max(0, item.startTime)
    item.duration = max(0, item.duration)
    textItems.append(item)
    return item
  }

  public mutating func setTextEnabled(_ id: UUID, isEnabled: Bool) {
    guard let index = textItems.firstIndex(where: { $0.id == id }) else { return }
    textItems[index].isEnabled = isEnabled
  }

  public mutating func setTextTrim(_ id: UUID, _ trim: TextTrim) {
    guard let index = textItems.firstIndex(where: { $0.id == id }) else { return }
    textItems[index].startTime = max(0, trim.startTime)
    textItems[index].duration = max(0, trim.duration)
  }

  public mutating func setTextStart(_ id: UUID, startTime: TimeInterval) {
    guard let index = textItems.firstIndex(where: { $0.id == id }) else { return }
    textItems[index].startTime = max(0, startTime)
  }

  public mutating func setTextStyle(_ id: UUID, _ style: TextStyle) {
    guard let index = textItems.firstIndex(where: { $0.id == id }) else { return }
    textItems[index].style = style
  }

  public mutating func setTextContent(_ id: UUID, _ text: String) {
    guard let index = textItems.firstIndex(where: { $0.id == id }) else { return }
    textItems[index].text = text
  }

  public mutating func setTextTransform(_ id: UUID, _ transform: CanvasObjectTransform) {
    guard let index = textItems.firstIndex(where: { $0.id == id }) else { return }
    textItems[index].canvasTransform = transform
  }

  public mutating func resetTextTransform(_ id: UUID) {
    guard let index = textItems.firstIndex(where: { $0.id == id }) else { return }
    textItems[index].canvasTransform = .identity
  }

  @discardableResult
  public mutating func duplicateText(_ id: UUID) -> TextItem? {
    guard let index = textItems.firstIndex(where: { $0.id == id }) else { return nil }
    let item = textItems[index]
    let copy = TextItem(
      startTime: item.endTime,
      duration: item.duration,
      isEnabled: item.isEnabled,
      text: item.text,
      style: item.style,
      canvasTransform: item.canvasTransform
    )
    textItems.insert(copy, at: index + 1)
    return copy
  }

  public func canSplitText(_ id: UUID, at time: TimeInterval, framesPerSecond: Double) -> Bool {
    guard let item = textItems.first(where: { $0.id == id }) else { return false }
    return TimelineSplit.canSplit(
      startTime: item.startTime,
      duration: item.duration,
      at: time,
      framesPerSecond: framesPerSecond
    )
  }

  @discardableResult
  public mutating func splitText(
    _ id: UUID,
    at time: TimeInterval,
    framesPerSecond: Double
  ) -> (left: UUID, right: UUID)? {
    guard let index = textItems.firstIndex(where: { $0.id == id }) else { return nil }
    let item = textItems[index]
    guard
      TimelineSplit.canSplit(
        startTime: item.startTime,
        duration: item.duration,
        at: time,
        framesPerSecond: framesPerSecond
      )
    else {
      return nil
    }
    let leftDuration = time - item.startTime
    textItems[index].duration = leftDuration
    let right = TextItem(
      startTime: time,
      duration: item.duration - leftDuration,
      isEnabled: item.isEnabled,
      text: item.text,
      style: item.style,
      canvasTransform: item.canvasTransform
    )
    textItems.insert(right, at: index + 1)
    return (item.id, right.id)
  }

  public mutating func removeText(_ id: UUID) {
    textItems.removeAll { $0.id == id }
  }
}

public enum TimelineReorderMath: Sendable {
  /// Index of `dropTime` among `others`, using each item's midpoint.
  /// The result is the destination after the moving item is removed.
  public static func destinationIndex(
    dropTime: TimeInterval,
    others: [(start: TimeInterval, duration: TimeInterval)]
  ) -> Int {
    var dest = others.count
    for (index, item) in others.enumerated() {
      if dropTime < item.start + item.duration / 2 {
        dest = index
        break
      }
    }
    return dest
  }

  /// Time of the insertion caret among remaining clips. Destination `0` is the
  /// 0s mark, not the first remaining clip's start — that clip may sit later.
  public static func insertionTime(
    dest: Int,
    others: [(start: TimeInterval, duration: TimeInterval)]
  ) -> TimeInterval {
    if dest <= 0 { return 0 }
    if dest >= others.count {
      return others.last.map { $0.start + $0.duration } ?? 0
    }
    return others[dest].start
  }

  public static func audioOrder(_ lhs: AudioItem, _ rhs: AudioItem) -> Bool {
    if lhs.startTime != rhs.startTime { return lhs.startTime < rhs.startTime }
    return lhs.id.uuidString < rhs.id.uuidString
  }
}

/// Playhead behavior when a timeline clip is selected.
public enum TimelineSelection: Sendable {
  /// Time to seek to after selecting a clip. `nil` when the playhead already
  /// sits inside `[start, start + duration)`.
  public static func seekTime(
    playhead: TimeInterval,
    start: TimeInterval,
    duration: TimeInterval
  ) -> TimeInterval? {
    let end = start + max(0, duration)
    if playhead >= start && playhead < end {
      return nil
    }
    return start
  }
}

public enum TimelineSplit: Sendable {
  public static func canSplit(
    startTime: TimeInterval,
    duration: TimeInterval,
    at time: TimeInterval,
    framesPerSecond: Double
  ) -> Bool {
    let minimum = VisualTrimMath.minimumDuration(framesPerSecond: framesPerSecond)
    let endTime = startTime + duration
    return time >= startTime + minimum && time <= endTime - minimum
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
