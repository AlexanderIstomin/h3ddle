import Foundation

public enum InterchangeProjection: Sendable {
  public static func document(
    from project: H3ddleProject,
    revision: Int = 0,
    extras: [String: JSONValue] = [:],
    mediaLocator: (AssetReference) -> String = defaultMediaLocator
  ) throws -> InterchangeDocument {
    let projectID = project.id.interchangeID
    var compositions: [InterchangeComposition] = []
    var transitions: [InterchangeTransition] = []
    let visualClips = try visualTrack(
      project: project,
      compositions: &compositions,
      transitions: &transitions
    )
    let audioClips = try audioTrack(project: project)
    let textClips = textTrack(
      project: project,
      compositions: &compositions
    )
    let sequenceDuration = max(
      project.timeline.visualDuration,
      project.timeline.audioTrackEnd,
      project.timeline.textTrackEnd
    )
    let sequence = InterchangeSequence(
      id: "\(projectID)-main",
      name: "Main",
      duration: sequenceDuration,
      tracks: [
        InterchangeTrack(
          id: "\(projectID)-v1",
          name: "V1",
          kind: "visual",
          clips: visualClips
        ),
        InterchangeTrack(
          id: "\(projectID)-t1",
          name: "T1",
          kind: "visual",
          clips: textClips
        ),
        InterchangeTrack(
          id: "\(projectID)-a1",
          name: "A1",
          kind: "audio",
          clips: audioClips
        ),
      ],
      transitions: transitions
    )
    return InterchangeDocument(
      id: projectID,
      revision: revision,
      name: project.name,
      settings: interchangeSettings(project.settings),
      assets: project.assets.map { interchangeAsset($0, mediaLocator: mediaLocator) },
      sequences: [sequence],
      compositions: compositions,
      extras: extras.filter { !InterchangeDocument.knownKeys.contains($0.key) }
    )
  }

  public static func project(
    from document: InterchangeDocument,
    resolvingMedia: (InterchangeAsset) throws -> URL
  ) throws -> H3ddleProject {
    let id = UUID(uuidString: document.id) ?? UUID()
    var assets: [AssetReference] = []
    for asset in document.assets {
      assets.append(
        try projectAsset(asset, url: resolvingMedia(asset))
      )
    }
    var timeline = ProjectTimeline()
    let compositions = Dictionary(
      uniqueKeysWithValues: document.compositions.map { ($0.id, $0) }
    )
    if let sequence = document.sequences.first {
      let visualTracks = sequence.tracks.filter {
        $0.kind == "visual" && $0.name != "T1"
      }
      let textTracks = sequence.tracks.filter { $0.name == "T1" }
      let audioTracks = sequence.tracks.filter { $0.kind == "audio" }
      timeline.importVisuals(
        visualTracks.flatMap(\.clips),
        transitions: sequence.transitions,
        compositions: compositions,
        assets: assets
      )
      timeline.importText(textTracks.flatMap(\.clips), compositions: compositions)
      timeline.importAudio(audioTracks.flatMap(\.clips), assets: assets)
    }
    return H3ddleProject(
      id: id,
      schemaVersion: H3ddleProject.currentSchemaVersion,
      name: document.name,
      assets: assets,
      timeline: timeline,
      settings: projectSettings(document.settings)
    )
  }

  public static func encode(
    _ document: InterchangeDocument
  ) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return try encoder.encode(document)
  }

  public static func decodeDocument(_ data: Data) throws -> InterchangeDocument {
    try JSONDecoder().decode(InterchangeDocument.self, from: data)
  }

  public static func defaultMediaLocator(_ asset: AssetReference) -> String {
    let ext = asset.url.pathExtension
    let file =
      ext.isEmpty
      ? asset.id.rawValue.interchangeID
      : "\(asset.id.rawValue.interchangeID).\(ext)"
    return "Media/\(file)"
  }
}

extension UUID {
  fileprivate var interchangeID: String { uuidString.lowercased() }
}

extension AssetID {
  fileprivate var interchangeID: String { rawValue.interchangeID }
}

private func interchangeAsset(
  _ asset: AssetReference,
  mediaLocator: (AssetReference) -> String
) -> InterchangeAsset {
  InterchangeAsset(
    id: asset.id.interchangeID,
    kind: asset.kind.rawValue,
    src: mediaLocator(asset),
    name: asset.displayName,
    duration: asset.duration
  )
}

private func projectAsset(_ asset: InterchangeAsset, url: URL) throws -> AssetReference {
  guard let kind = MediaKind(rawValue: asset.kind) else {
    throw InterchangeError.unexpectedType(asset.kind)
  }
  guard let id = UUID(uuidString: asset.id).map(AssetID.init(rawValue:)) else {
    throw InterchangeError.unexpectedType("asset.id")
  }
  return AssetReference(
    id: id,
    kind: kind,
    displayName: asset.name ?? url.deletingPathExtension().lastPathComponent,
    url: url,
    duration: max(0, asset.duration ?? 0)
  )
}

private func interchangeSettings(_ settings: ProjectSettings) -> InterchangeSettings {
  InterchangeSettings(
    width: settings.width,
    height: settings.height,
    fps: settings.framesPerSecond,
    backgroundColor: settings.background.rawValue,
    masterGain: settings.masterGain,
    toneMapping: interchangeToneMapping(settings.toneMapping),
    exposure: settings.exposure,
    target: settings.platform.rawValue
  )
}

private func projectSettings(_ settings: InterchangeSettings) -> ProjectSettings {
  var next = ProjectSettings(
    width: settings.width,
    height: settings.height,
    framesPerSecond: settings.fps,
    background: ProjectBackground(rawValue: settings.backgroundColor ?? "") ?? .black,
    platform: ProjectPlatform(rawValue: settings.target ?? "") ?? .custom,
    masterGain: settings.masterGain ?? 1,
    toneMapping: projectToneMapping(settings.toneMapping),
    exposure: settings.exposure ?? 1
  )
  if next.platform == .custom {
    next.platform = .custom
  }
  return next
}

private func interchangeToneMapping(_ value: ProjectToneMapping) -> String {
  switch value {
  case .none: "none"
  case .agx: "agx"
  case .aces: "aces-filmic"
  case .neutral: "neutral"
  }
}

private func projectToneMapping(_ value: String?) -> ProjectToneMapping {
  switch value {
  case "agx": .agx
  case "aces-filmic", "aces": .aces
  case "neutral": .neutral
  default: .none
  }
}

private func visualTrack(
  project: H3ddleProject,
  compositions: inout [InterchangeComposition],
  transitions: inout [InterchangeTransition]
) throws -> [InterchangeClip] {
  var clips: [InterchangeClip] = []
  let placements = project.timeline.visualPlacements
  for (index, placement) in placements.enumerated() {
    let item = placement.item
    guard let asset = project.asset(id: item.assetID) else {
      throw InterchangeError.missingAsset
    }
    let clipID = item.id.interchangeID
    let compositionID = "\(clipID)-comp"
    compositions.append(
      mediaComposition(
        id: compositionID,
        name: asset.displayName,
        duration: item.duration,
        asset: asset,
        item: item,
        settings: project.settings
      )
    )
    if let transition = item.transition, index > 0 {
      let outgoing = placements[index - 1].item
      transitions.append(
        InterchangeTransition(
          id: "\(clipID)-transition",
          fromClipId: outgoing.id.interchangeID,
          toClipId: clipID,
          kind: interchangeTransitionKind(transition.kind),
          duration: transition.duration
        )
      )
    }
    clips.append(
      InterchangeClip(
        id: clipID,
        timelineIn: placement.startTime,
        timelineOut: placement.startTime + item.duration,
        sourceIn: asset.kind == .video ? item.sourceOffset : 0,
        compositionId: compositionID,
        assetId: asset.kind == .video ? asset.id.interchangeID : nil,
        enabled: item.isEnabled ? nil : false,
        gain: asset.kind == .video && !item.includesNativeAudio ? 0 : nil,
        effects: item.effects.map(interchangeEffect)
      )
    )
  }
  return clips
}

private func audioTrack(
  project: H3ddleProject
) throws -> [InterchangeClip] {
  return try project.timeline.audioItems.map { item in
    guard project.asset(id: item.assetID) != nil else {
      throw InterchangeError.missingAsset
    }
    return InterchangeClip(
      id: item.id.interchangeID,
      timelineIn: item.startTime,
      timelineOut: item.endTime,
      sourceIn: item.sourceOffset,
      assetId: item.assetID.interchangeID,
      enabled: item.isEnabled ? nil : false,
      gain: item.gain == 1 ? nil : Double(item.gain)
    )
  }
}

private func textTrack(
  project: H3ddleProject,
  compositions: inout [InterchangeComposition]
) -> [InterchangeClip] {
  project.timeline.textItems.map { item in
    let clipID = item.id.interchangeID
    let compositionID = "\(clipID)-comp"
    compositions.append(
      textComposition(
        id: compositionID,
        item: item,
        settings: project.settings
      )
    )
    return InterchangeClip(
      id: clipID,
      timelineIn: item.startTime,
      timelineOut: item.endTime,
      compositionId: compositionID,
      enabled: item.isEnabled ? nil : false
    )
  }
}

private func mediaComposition(
  id: String,
  name: String,
  duration: TimeInterval,
  asset: AssetReference,
  item: VisualItem,
  settings: ProjectSettings
) -> InterchangeComposition {
  let layerKind = asset.kind == .image ? "image" : "video"
  var layer = InterchangeLayer(
    id: "\(item.id.interchangeID)-layer",
    kind: layerKind,
    transform: interchangeTransform(item.canvasTransform, settings: settings),
    assetId: asset.id.interchangeID,
    canvasFit: item.canvasFit.rawValue,
    canvasTranslationX: item.translationX,
    canvasTranslationY: item.translationY,
    canvasScale: item.uniformScale,
    canvasRotationRadians: item.rotationRadians
  )
  if asset.kind == .video {
    layer.timing = InterchangeSurfaceTiming(
      timelineIn: 0,
      timelineOut: duration,
      sourceIn: item.sourceOffset
    )
  }
  return InterchangeComposition(
    id: id,
    name: name,
    duration: duration,
    spatial: .planar(width: settings.width, height: settings.height),
    layers: [layer]
  )
}

private func textComposition(
  id: String,
  item: TextItem,
  settings: ProjectSettings
) -> InterchangeComposition {
  let layer = InterchangeLayer(
    id: "\(item.id.interchangeID)-layer",
    kind: "text",
    transform: interchangeTransform(item.canvasTransform, settings: settings),
    text: interchangeText(item),
    canvasFit: item.canvasTransform.fit.rawValue,
    canvasTranslationX: item.canvasTransform.translationX,
    canvasTranslationY: item.canvasTransform.translationY,
    canvasScale: item.canvasTransform.scale,
    canvasRotationRadians: item.canvasTransform.rotationRadians
  )
  return InterchangeComposition(
    id: id,
    name: item.text,
    duration: item.duration,
    spatial: .planar(width: settings.width, height: settings.height),
    layers: [layer]
  )
}

private func interchangeTransform(
  _ transform: CanvasObjectTransform,
  settings: ProjectSettings
) -> InterchangeTransform {
  InterchangeTransform(
    position: [
      transform.translationX * Double(settings.width),
      transform.translationY * Double(settings.height),
      1,
    ],
    rotation: [0, 0, transform.rotationRadians],
    scale: [transform.scale, transform.scale, 1]
  )
}

private func canvasTransform(from layer: InterchangeLayer) -> CanvasObjectTransform {
  if layer.canvasTranslationX != nil || layer.canvasFit != nil {
    return CanvasObjectTransform(
      fit: CanvasFit(rawValue: layer.canvasFit ?? "") ?? .fit,
      translationX: layer.canvasTranslationX ?? 0,
      translationY: layer.canvasTranslationY ?? 0,
      scale: layer.canvasScale ?? 1,
      rotationRadians: layer.canvasRotationRadians ?? 0
    )
  }
  return CanvasObjectTransform(
    fit: .fit,
    translationX: 0,
    translationY: 0,
    scale: layer.transform.scale.first ?? 1,
    rotationRadians: layer.transform.rotation[safe: 2] ?? 0
  )
}

private func interchangeEffect(_ effect: VisualEffectInstance) -> InterchangeEffect {
  InterchangeEffect(
    id: effect.id.uuidString.lowercased(),
    defId: interchangeEffectID(effect.kind),
    enabled: effect.isEnabled,
    params: effect.parameters
  )
}

private func projectEffect(_ effect: InterchangeEffect) -> VisualEffectInstance? {
  guard let kind = projectEffectKind(effect.defId) else { return nil }
  let id = UUID(uuidString: effect.id) ?? UUID()
  return VisualEffectInstance(
    id: id,
    kind: kind,
    isEnabled: effect.enabled,
    parameters: effect.params
  )
}

private func interchangeEffectID(_ kind: VisualEffectKind) -> String {
  switch kind {
  case .colorGrade: "color.grade"
  case .vignette: "stylize.vignette"
  case .filmGrain: "stylize.filmGrain"
  case .sharpen: "stylize.sharpen"
  case .blur: "blur.gaussian"
  case .bloom: "stylize.bloom"
  case .chromaKey: "key.chroma"
  }
}

private func projectEffectKind(_ defId: String) -> VisualEffectKind? {
  switch defId {
  case "color.grade": .colorGrade
  case "stylize.vignette": .vignette
  case "stylize.filmGrain": .filmGrain
  case "stylize.sharpen": .sharpen
  case "blur.gaussian": .blur
  case "stylize.bloom": .bloom
  case "key.chroma": .chromaKey
  default: nil
  }
}

private func interchangeTransitionKind(_ kind: VisualTransitionKind) -> String {
  switch kind {
  case .dissolve: "cross-dissolve"
  case .fade: "dip-black"
  case .wipe: "wipe"
  }
}

private func projectTransitionKind(_ kind: String) -> VisualTransitionKind? {
  switch kind {
  case "cross-dissolve", "dissolve": .dissolve
  case "dip-black", "fade": .fade
  case "wipe": .wipe
  default: nil
  }
}

private func interchangeText(_ item: TextItem) -> InterchangeTextContent {
  let style = item.style
  let fill = rgba(style.fill)
  return InterchangeTextContent(
    value: item.text,
    engine: InterchangeTextEngine(
      face: InterchangeFontFace(
        faceId: style.fontFamily,
        sourceRevision: style.fontPostScriptName ?? style.fontFamily,
        weight: style.fontWeight,
        style: style.italic ? "italic" : "normal"
      ),
      render: InterchangeTextRender(
        fill: fill,
        strokeWidth: style.strokeWidth,
        strokeColor: rgba(style.strokeColor),
        backgroundColor: rgba(style.backgroundColor),
        backgroundPadding: style.backgroundPadding,
        backgroundRadius: style.backgroundCornerRadius,
        shadowColor: rgba(style.shadowColor),
        shadowBlur: style.shadowBlur,
        shadowOffset: [style.shadowOffsetX, style.shadowOffsetY]
      ),
      layout: InterchangeTextLayout(
        fontSize: style.fontSize,
        lineHeight: style.lineHeight,
        letterSpacing: style.letterSpacing,
        align: interchangeAlign(style.alignment),
        wrap: style.wrap == .wrap ? "word" : "none",
        boxKind: style.boxWidth == nil ? "auto" : "fixed",
        boxWidth: style.boxWidth
      )
    )
  )
}

private func projectTextStyle(_ content: InterchangeTextContent) -> TextStyle {
  let render = content.engine.render
  let layout = content.engine.layout
  let face = content.engine.face
  return TextStyle(
    fontFamily: face.faceId,
    fontPostScriptName: face.sourceRevision == face.faceId ? nil : face.sourceRevision,
    fontWeight: face.weight,
    italic: face.style == "italic" || face.style == "oblique",
    fontSize: layout.fontSize,
    alignment: projectAlign(layout.align),
    fill: color(render.fill),
    wrap: layout.wrap == "word" ? .wrap : .none,
    boxWidth: layout.boxKind == "fixed" ? layout.boxWidth : nil,
    lineHeight: layout.lineHeight,
    letterSpacing: layout.letterSpacing,
    strokeWidth: render.strokeWidth,
    strokeColor: color(render.strokeColor),
    shadowOffsetX: render.shadowOffset[safe: 0] ?? 0,
    shadowOffsetY: render.shadowOffset[safe: 1] ?? 0,
    shadowBlur: render.shadowBlur,
    shadowColor: color(render.shadowColor),
    backgroundColor: color(render.backgroundColor),
    backgroundPadding: render.backgroundPadding,
    backgroundCornerRadius: render.backgroundRadius
  )
}

private func interchangeAlign(_ alignment: TextAlignment) -> String {
  switch alignment {
  case .leading: "start"
  case .center: "center"
  case .trailing: "end"
  }
}

private func projectAlign(_ align: String) -> TextAlignment {
  switch align {
  case "start", "left": .leading
  case "end", "right": .trailing
  default: .center
  }
}

private func rgba(_ color: TextColor) -> [Double] {
  [color.r, color.g, color.b, color.a]
}

private func color(_ values: [Double]) -> TextColor {
  TextColor(
    r: values[safe: 0] ?? 0,
    g: values[safe: 1] ?? 0,
    b: values[safe: 2] ?? 0,
    a: values[safe: 3] ?? 1
  )
}

extension ProjectTimeline {
  fileprivate mutating func importVisuals(
    _ clips: [InterchangeClip],
    transitions: [InterchangeTransition],
    compositions: [String: InterchangeComposition],
    assets: [AssetReference]
  ) {
    let incoming = Dictionary(uniqueKeysWithValues: transitions.map { ($0.toClipId, $0) })
    let ordered = clips.sorted { $0.timelineIn < $1.timelineIn }
    var items: [VisualItem] = []
    var cursor: TimeInterval = 0
    for (index, clip) in ordered.enumerated() {
      let composition = clip.compositionId.flatMap { compositions[$0] }
      let layer = composition?.layers.first
      let assetID = resolvedAssetID(clip: clip, layer: layer)
      guard let assetID, assets.contains(where: { $0.id == assetID }) else { continue }
      let duration = clip.duration
      let overlap: TimeInterval
      let transition: VisualTransition?
      if let incoming = incoming[clip.id],
        let kind = projectTransitionKind(incoming.kind),
        index > 0
      {
        overlap = incoming.duration
        transition = VisualTransition(kind: kind, duration: incoming.duration)
      } else {
        overlap = 0
        transition = nil
      }
      let expectedStart = max(0, cursor - overlap)
      let gapBefore = max(0, clip.timelineIn - expectedStart)
      let canvas = layer.map(canvasTransform(from:)) ?? .identity
      let asset = assets.first { $0.id == assetID }
      items.append(
        VisualItem(
          id: UUID(uuidString: clip.id) ?? UUID(),
          assetID: assetID,
          duration: duration,
          isEnabled: clip.enabled ?? true,
          includesNativeAudio: asset?.kind == .video && (clip.gain ?? 1) > 0.000_1,
          sourceOffset: clip.sourceIn,
          gapBefore: gapBefore,
          canvasFit: canvas.fit,
          rotationTurns: canvas.rotationTurns,
          translationX: canvas.translationX,
          translationY: canvas.translationY,
          uniformScale: canvas.scale,
          rotationRadians: canvas.rotationRadians,
          transition: transition,
          effects: clip.effects.compactMap(projectEffect)
        )
      )
      cursor = clip.timelineIn + duration
    }
    self = ProjectTimeline(
      visualItems: items,
      audioItems: audioItems,
      textItems: textItems
    )
  }

  fileprivate mutating func importAudio(
    _ clips: [InterchangeClip],
    assets: [AssetReference]
  ) {
    let items: [AudioItem] = clips.compactMap { clip in
      guard let raw = clip.assetId, let assetID = AssetID(uuidString: raw),
        assets.contains(where: { $0.id == assetID })
      else { return nil }
      return AudioItem(
        id: UUID(uuidString: clip.id) ?? UUID(),
        assetID: assetID,
        startTime: clip.timelineIn,
        duration: clip.duration,
        isEnabled: clip.enabled ?? true,
        gain: Float(clip.gain ?? 1),
        sourceOffset: clip.sourceIn
      )
    }
    self = ProjectTimeline(
      visualItems: visualItems,
      audioItems: items,
      textItems: textItems
    )
  }

  fileprivate mutating func importText(
    _ clips: [InterchangeClip],
    compositions: [String: InterchangeComposition]
  ) {
    let items: [TextItem] = clips.compactMap { clip in
      let layer = clip.compositionId.flatMap { compositions[$0] }?.layers.first
      guard let text = layer?.text else { return nil }
      return TextItem(
        id: UUID(uuidString: clip.id) ?? UUID(),
        startTime: clip.timelineIn,
        duration: clip.duration,
        isEnabled: clip.enabled ?? true,
        text: text.value,
        style: projectTextStyle(text),
        canvasTransform: layer.map(canvasTransform(from:)) ?? .identity
      )
    }
    self = ProjectTimeline(
      visualItems: visualItems,
      audioItems: audioItems,
      textItems: items
    )
  }
}

private func resolvedAssetID(clip: InterchangeClip, layer: InterchangeLayer?) -> AssetID? {
  if let raw = clip.assetId ?? layer?.assetId {
    return AssetID(uuidString: raw)
  }
  return nil
}
