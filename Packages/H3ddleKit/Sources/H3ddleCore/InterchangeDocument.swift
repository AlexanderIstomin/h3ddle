import Foundation

/// Editorial interchange document. Independently specified; unknown keys round-trip.
public struct InterchangeDocument: Hashable, Sendable {
  public static let currentSchemaVersion = 1

  public var schemaVersion: Int
  public var id: String
  public var revision: Int
  public var name: String
  public var settings: InterchangeSettings
  public var assets: [InterchangeAsset]
  public var sequences: [InterchangeSequence]
  public var compositions: [InterchangeComposition]
  public var extras: [String: JSONValue]

  public init(
    schemaVersion: Int = InterchangeDocument.currentSchemaVersion,
    id: String,
    revision: Int = 0,
    name: String = "Untitled Project",
    settings: InterchangeSettings,
    assets: [InterchangeAsset] = [],
    sequences: [InterchangeSequence] = [],
    compositions: [InterchangeComposition] = [],
    extras: [String: JSONValue] = [:]
  ) {
    self.schemaVersion = schemaVersion
    self.id = id
    self.revision = revision
    self.name = name
    self.settings = settings
    self.assets = assets
    self.sequences = sequences
    self.compositions = compositions
    self.extras = extras
  }

  static let knownKeys: Set<String> = [
    "schemaVersion", "id", "revision", "name", "settings", "assets", "sequences",
    "compositions",
  ]
}
public struct InterchangeSettings: Hashable, Sendable {
  public var width: Int
  public var height: Int
  public var fps: Double
  public var colorSpace: String
  public var alphaMode: String
  public var backgroundColor: String?
  public var masterGain: Double?
  public var toneMapping: String?
  public var exposure: Double?
  public var target: String?
  public var extras: [String: JSONValue]

  public init(
    width: Int,
    height: Int,
    fps: Double,
    colorSpace: String = "srgb",
    alphaMode: String = "premultiplied",
    backgroundColor: String? = nil,
    masterGain: Double? = nil,
    toneMapping: String? = nil,
    exposure: Double? = nil,
    target: String? = nil,
    extras: [String: JSONValue] = [:]
  ) {
    self.width = width
    self.height = height
    self.fps = fps
    self.colorSpace = colorSpace
    self.alphaMode = alphaMode
    self.backgroundColor = backgroundColor
    self.masterGain = masterGain
    self.toneMapping = toneMapping
    self.exposure = exposure
    self.target = target
    self.extras = extras
  }

  static let knownKeys: Set<String> = [
    "width", "height", "fps", "colorSpace", "alphaMode", "backgroundColor",
    "masterGain", "toneMapping", "exposure", "target",
  ]
}

public struct InterchangeAsset: Hashable, Sendable {
  public var id: String
  public var kind: String
  public var src: String
  public var name: String?
  public var duration: Double?
  public var extras: [String: JSONValue]

  public init(
    id: String,
    kind: String,
    src: String,
    name: String? = nil,
    duration: Double? = nil,
    extras: [String: JSONValue] = [:]
  ) {
    self.id = id
    self.kind = kind
    self.src = src
    self.name = name
    self.duration = duration
    self.extras = extras
  }

  static let knownKeys: Set<String> = ["id", "kind", "src", "name", "duration"]
}

public struct InterchangeSequence: Hashable, Sendable {
  public var id: String
  public var name: String?
  public var duration: Double
  public var tracks: [InterchangeTrack]
  public var transitions: [InterchangeTransition]
  public var extras: [String: JSONValue]

  public init(
    id: String,
    name: String? = "Main",
    duration: Double = 0,
    tracks: [InterchangeTrack] = [],
    transitions: [InterchangeTransition] = [],
    extras: [String: JSONValue] = [:]
  ) {
    self.id = id
    self.name = name
    self.duration = duration
    self.tracks = tracks
    self.transitions = transitions
    self.extras = extras
  }

  static let knownKeys: Set<String> = [
    "id", "name", "duration", "tracks", "transitions",
  ]
}

public struct InterchangeTrack: Hashable, Sendable {
  public var id: String
  public var name: String?
  public var kind: String
  public var clips: [InterchangeClip]
  public var extras: [String: JSONValue]

  public init(
    id: String,
    name: String? = nil,
    kind: String,
    clips: [InterchangeClip] = [],
    extras: [String: JSONValue] = [:]
  ) {
    self.id = id
    self.name = name
    self.kind = kind
    self.clips = clips
    self.extras = extras
  }

  static let knownKeys: Set<String> = ["id", "name", "kind", "clips"]
}

public struct InterchangeClip: Hashable, Sendable {
  public var id: String
  public var timelineIn: Double
  public var timelineOut: Double
  public var sourceIn: Double
  public var playbackRate: Double
  public var compositionId: String?
  public var assetId: String?
  public var enabled: Bool?
  public var gain: Double?
  public var effects: [InterchangeEffect]
  public var extras: [String: JSONValue]

  public init(
    id: String,
    timelineIn: Double,
    timelineOut: Double,
    sourceIn: Double = 0,
    playbackRate: Double = 1,
    compositionId: String? = nil,
    assetId: String? = nil,
    enabled: Bool? = nil,
    gain: Double? = nil,
    effects: [InterchangeEffect] = [],
    extras: [String: JSONValue] = [:]
  ) {
    self.id = id
    self.timelineIn = timelineIn
    self.timelineOut = timelineOut
    self.sourceIn = sourceIn
    self.playbackRate = playbackRate
    self.compositionId = compositionId
    self.assetId = assetId
    self.enabled = enabled
    self.gain = gain
    self.effects = effects
    self.extras = extras
  }

  public var duration: Double { max(0, timelineOut - timelineIn) }

  static let knownKeys: Set<String> = [
    "id", "timelineIn", "timelineOut", "sourceIn", "playbackRate", "compositionId",
    "assetId", "enabled", "gain", "effects",
  ]
}

public struct InterchangeTransition: Hashable, Sendable {
  public var id: String
  public var fromClipId: String
  public var toClipId: String
  public var kind: String
  public var duration: Double
  public var easing: String
  public var extras: [String: JSONValue]

  public init(
    id: String,
    fromClipId: String,
    toClipId: String,
    kind: String,
    duration: Double,
    easing: String = "linear",
    extras: [String: JSONValue] = [:]
  ) {
    self.id = id
    self.fromClipId = fromClipId
    self.toClipId = toClipId
    self.kind = kind
    self.duration = duration
    self.easing = easing
    self.extras = extras
  }

  static let knownKeys: Set<String> = [
    "id", "fromClipId", "toClipId", "kind", "duration", "easing",
  ]
}

public struct InterchangeEffect: Hashable, Sendable {
  public var id: String
  public var defId: String
  public var version: Int
  public var enabled: Bool
  public var params: [String: Double]
  public var extras: [String: JSONValue]

  public init(
    id: String,
    defId: String,
    version: Int = 1,
    enabled: Bool = true,
    params: [String: Double] = [:],
    extras: [String: JSONValue] = [:]
  ) {
    self.id = id
    self.defId = defId
    self.version = version
    self.enabled = enabled
    self.params = params
    self.extras = extras
  }

  static let knownKeys: Set<String> = ["id", "defId", "version", "enabled", "params"]
}

public struct InterchangeComposition: Hashable, Sendable {
  public var id: String
  public var name: String?
  public var duration: Double
  public var spatial: InterchangeSpatial
  public var layers: [InterchangeLayer]
  public var extras: [String: JSONValue]

  public init(
    id: String,
    name: String? = nil,
    duration: Double,
    spatial: InterchangeSpatial,
    layers: [InterchangeLayer] = [],
    extras: [String: JSONValue] = [:]
  ) {
    self.id = id
    self.name = name
    self.duration = duration
    self.spatial = spatial
    self.layers = layers
    self.extras = extras
  }

  static let knownKeys: Set<String> = ["id", "name", "duration", "spatial", "layers"]
}

public struct InterchangeSpatial: Hashable, Sendable {
  public var version: Int
  public var programCamera: InterchangeCamera
  public var environment: InterchangeEnvironment
  public var lights: [JSONValue]
  public var exposure: Double
  public var toneMapping: String
  public var extras: [String: JSONValue]

  public init(
    version: Int = 1,
    programCamera: InterchangeCamera,
    environment: InterchangeEnvironment = InterchangeEnvironment(),
    lights: [JSONValue] = [],
    exposure: Double = 1,
    toneMapping: String = "none",
    extras: [String: JSONValue] = [:]
  ) {
    self.version = version
    self.programCamera = programCamera
    self.environment = environment
    self.lights = lights
    self.exposure = exposure
    self.toneMapping = toneMapping
    self.extras = extras
  }

  public static func planar(width: Int, height: Int) -> InterchangeSpatial {
    InterchangeSpatial(
      programCamera: InterchangeCamera(width: width, height: height)
    )
  }

  static let knownKeys: Set<String> = [
    "version", "programCamera", "environment", "lights", "exposure", "toneMapping",
  ]
}

public struct InterchangeCamera: Hashable, Sendable {
  public var kind: String
  public var width: Double
  public var height: Double
  public var near: Double
  public var far: Double
  public var position: [Double]
  public var target: [Double]
  public var up: [Double]
  public var extras: [String: JSONValue]

  public init(
    kind: String = "orthographic",
    width: Int,
    height: Int,
    near: Double = 0.1,
    far: Double = 10_000,
    position: [Double] = [0, 0, 1_000],
    target: [Double] = [0, 0, 0],
    up: [Double] = [0, 1, 0],
    extras: [String: JSONValue] = [:]
  ) {
    self.kind = kind
    self.width = Double(width)
    self.height = Double(height)
    self.near = near
    self.far = far
    self.position = position
    self.target = target
    self.up = up
    self.extras = extras
  }

  static let knownKeys: Set<String> = [
    "kind", "width", "height", "near", "far", "position", "target", "up",
  ]
}

public struct InterchangeEnvironment: Hashable, Sendable {
  public var kind: String
  public var extras: [String: JSONValue]

  public init(kind: String = "none", extras: [String: JSONValue] = [:]) {
    self.kind = kind
    self.extras = extras
  }

  static let knownKeys: Set<String> = ["kind"]
}

public struct InterchangeTransform: Hashable, Sendable {
  public var position: [Double]
  public var rotation: [Double]
  public var scale: [Double]
  public var anchor: [Double]

  public static let identity = InterchangeTransform()

  public init(
    position: [Double] = [0, 0, 1],
    rotation: [Double] = [0, 0, 0],
    scale: [Double] = [1, 1, 1],
    anchor: [Double] = [0, 0, 0]
  ) {
    self.position = position
    self.rotation = rotation
    self.scale = scale
    self.anchor = anchor
  }
}

public struct InterchangeSurfaceTiming: Hashable, Sendable {
  public var timelineIn: Double
  public var timelineOut: Double
  public var sourceIn: Double
  public var playbackRate: Double

  public init(
    timelineIn: Double,
    timelineOut: Double,
    sourceIn: Double = 0,
    playbackRate: Double = 1
  ) {
    self.timelineIn = timelineIn
    self.timelineOut = timelineOut
    self.sourceIn = sourceIn
    self.playbackRate = playbackRate
  }
}

public struct InterchangeLayer: Hashable, Sendable {
  public var id: String
  public var kind: String
  public var spatial: String
  public var transform: InterchangeTransform
  public var opacity: Double
  public var blendMode: String
  public var visible: Bool
  public var effects: [InterchangeEffect]
  public var assetId: String?
  public var timing: InterchangeSurfaceTiming?
  public var text: InterchangeTextContent?
  public var canvasFit: String?
  public var canvasTranslationX: Double?
  public var canvasTranslationY: Double?
  public var canvasScale: Double?
  public var canvasRotationRadians: Double?
  public var extras: [String: JSONValue]

  public init(
    id: String,
    kind: String,
    spatial: String = "2d",
    transform: InterchangeTransform = .identity,
    opacity: Double = 1,
    blendMode: String = "normal",
    visible: Bool = true,
    effects: [InterchangeEffect] = [],
    assetId: String? = nil,
    timing: InterchangeSurfaceTiming? = nil,
    text: InterchangeTextContent? = nil,
    canvasFit: String? = nil,
    canvasTranslationX: Double? = nil,
    canvasTranslationY: Double? = nil,
    canvasScale: Double? = nil,
    canvasRotationRadians: Double? = nil,
    extras: [String: JSONValue] = [:]
  ) {
    self.id = id
    self.kind = kind
    self.spatial = spatial
    self.transform = transform
    self.opacity = opacity
    self.blendMode = blendMode
    self.visible = visible
    self.effects = effects
    self.assetId = assetId
    self.timing = timing
    self.text = text
    self.canvasFit = canvasFit
    self.canvasTranslationX = canvasTranslationX
    self.canvasTranslationY = canvasTranslationY
    self.canvasScale = canvasScale
    self.canvasRotationRadians = canvasRotationRadians
    self.extras = extras
  }

  static let knownKeys: Set<String> = [
    "id", "kind", "spatial", "transform", "opacity", "blendMode", "visible", "effects",
    "assetId", "timing", "text", "canvasFit", "canvasTranslationX", "canvasTranslationY",
    "canvasScale", "canvasRotationRadians",
  ]
}

public struct InterchangeTextContent: Hashable, Sendable {
  public var value: String
  public var engine: InterchangeTextEngine
  public var spans: [JSONValue]
  public var timedWords: [JSONValue]
  public var extras: [String: JSONValue]

  public init(
    value: String,
    engine: InterchangeTextEngine,
    spans: [JSONValue] = [],
    timedWords: [JSONValue] = [],
    extras: [String: JSONValue] = [:]
  ) {
    self.value = value
    self.engine = engine
    self.spans = spans
    self.timedWords = timedWords
    self.extras = extras
  }

  static let knownKeys: Set<String> = ["value", "engine", "spans", "timedWords"]
}

public struct InterchangeTextEngine: Hashable, Sendable {
  public var version: Int
  public var semanticRevision: String
  public var layoutEngineRevision: String
  public var renderContractRevision: String
  public var face: InterchangeFontFace
  public var fallbackFaces: [InterchangeFontFace]
  public var render: InterchangeTextRender
  public var layout: InterchangeTextLayout
  public var animatorStack: JSONValue
  public var extras: [String: JSONValue]

  public init(
    version: Int = 1,
    semanticRevision: String = "h3ddle-text-v1",
    layoutEngineRevision: String = "coretext",
    renderContractRevision: String = "h3ddle-flat-text-v1",
    face: InterchangeFontFace,
    fallbackFaces: [InterchangeFontFace] = [],
    render: InterchangeTextRender,
    layout: InterchangeTextLayout,
    animatorStack: JSONValue = .object([
      "version": .int(1),
      "animators": .array([]),
    ]),
    extras: [String: JSONValue] = [:]
  ) {
    self.version = version
    self.semanticRevision = semanticRevision
    self.layoutEngineRevision = layoutEngineRevision
    self.renderContractRevision = renderContractRevision
    self.face = face
    self.fallbackFaces = fallbackFaces
    self.render = render
    self.layout = layout
    self.animatorStack = animatorStack
    self.extras = extras
  }

  static let knownKeys: Set<String> = [
    "version", "semanticRevision", "layoutEngineRevision", "renderContractRevision",
    "face", "fallbackFaces", "render", "layout", "animatorStack",
  ]
}

public struct InterchangeFontFace: Hashable, Sendable {
  public var version: Int
  public var sourceKind: String
  public var faceId: String
  public var sourceRevision: String
  public var faceIndex: Int
  public var weight: Int
  public var style: String
  public var stretch: Int
  public var extras: [String: JSONValue]

  public init(
    version: Int = 1,
    sourceKind: String = "builtin",
    faceId: String,
    sourceRevision: String,
    faceIndex: Int = 0,
    weight: Int = 400,
    style: String = "normal",
    stretch: Int = 100,
    extras: [String: JSONValue] = [:]
  ) {
    self.version = version
    self.sourceKind = sourceKind
    self.faceId = faceId
    self.sourceRevision = sourceRevision
    self.faceIndex = faceIndex
    self.weight = weight
    self.style = style
    self.stretch = stretch
    self.extras = extras
  }

  static let knownKeys: Set<String> = [
    "version", "source", "sourceRevision", "faceIndex", "weight", "style", "stretch",
    "axes",
  ]
}

public struct InterchangeTextRender: Hashable, Sendable {
  public var kind: String
  public var fill: [Double]
  public var strokeWidth: Double
  public var strokeColor: [Double]
  public var backgroundColor: [Double]
  public var backgroundPadding: Double
  public var backgroundRadius: Double
  public var shadowColor: [Double]
  public var shadowBlur: Double
  public var shadowOffset: [Double]

  public init(
    kind: String = "flat",
    fill: [Double],
    strokeWidth: Double = 0,
    strokeColor: [Double] = [0, 0, 0, 1],
    backgroundColor: [Double] = [0, 0, 0, 0],
    backgroundPadding: Double = 0,
    backgroundRadius: Double = 0,
    shadowColor: [Double] = [0, 0, 0, 0.5],
    shadowBlur: Double = 0,
    shadowOffset: [Double] = [0, 0]
  ) {
    self.kind = kind
    self.fill = fill
    self.strokeWidth = strokeWidth
    self.strokeColor = strokeColor
    self.backgroundColor = backgroundColor
    self.backgroundPadding = backgroundPadding
    self.backgroundRadius = backgroundRadius
    self.shadowColor = shadowColor
    self.shadowBlur = shadowBlur
    self.shadowOffset = shadowOffset
  }
}

public struct InterchangeTextLayout: Hashable, Sendable {
  public var version: Int
  public var fontSize: Double
  public var lineHeight: Double
  public var letterSpacing: Double
  public var language: String
  public var direction: String
  public var align: String
  public var wrap: String
  public var boxKind: String
  public var boxWidth: Double?
  public var extras: [String: JSONValue]

  public init(
    version: Int = 1,
    fontSize: Double,
    lineHeight: Double = 1.2,
    letterSpacing: Double = 0,
    language: String = "und",
    direction: String = "auto",
    align: String = "center",
    wrap: String = "none",
    boxKind: String = "auto",
    boxWidth: Double? = nil,
    extras: [String: JSONValue] = [:]
  ) {
    self.version = version
    self.fontSize = fontSize
    self.lineHeight = lineHeight
    self.letterSpacing = letterSpacing
    self.language = language
    self.direction = direction
    self.align = align
    self.wrap = wrap
    self.boxKind = boxKind
    self.boxWidth = boxWidth
    self.extras = extras
  }

  static let knownKeys: Set<String> = [
    "version", "fontSize", "lineHeight", "letterSpacing", "language", "direction",
    "features", "whiteSpace", "tabStops", "align", "box", "wrap", "overflow",
    "autoFit", "hyphenation",
  ]
}

extension InterchangeDocument: Codable {
  public init(from decoder: any Decoder) throws {
    try self.init(JSONValue(from: decoder))
  }

  public func encode(to encoder: any Encoder) throws {
    try json.encode(to: encoder)
  }

  public init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    let version = try object.requiredInt("schemaVersion")
    if version > InterchangeDocument.currentSchemaVersion {
      throw InterchangeError.unsupportedSchemaVersion(version)
    }
    schemaVersion = version
    id = try object.requiredString("id")
    revision = object.int("revision", default: 0)
    name = object.string("name") ?? "Untitled Project"
    settings = try InterchangeSettings(object.object("settings").json)
    assets = try decodeArray(object.array("assets"), InterchangeAsset.init)
    sequences = try decodeArray(object.array("sequences"), InterchangeSequence.init)
    compositions = try decodeArray(
      object.array("compositions"),
      InterchangeComposition.init
    )
    extras = object.extras(excluding: Self.knownKeys)
  }

  public var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("schemaVersion", schemaVersion)
    object.set("id", id)
    object.set("revision", revision)
    object.set("name", name)
    object.set("settings", settings.json)
    object.set("assets", jsonArray(assets, \.json))
    object.set("sequences", jsonArray(sequences, \.json))
    object.set("compositions", jsonArray(compositions, \.json))
    return object.json
  }
}

extension InterchangeSettings {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    width = try object.requiredInt("width")
    height = try object.requiredInt("height")
    fps = try object.requiredDouble("fps")
    colorSpace = object.string("colorSpace") ?? "srgb"
    alphaMode = object.string("alphaMode") ?? "premultiplied"
    backgroundColor = object.string("backgroundColor")
    masterGain = object.double("masterGain")
    toneMapping = object.string("toneMapping")
    exposure = object.double("exposure")
    target = object.string("target")
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("width", width)
    object.set("height", height)
    object.set("fps", fps)
    object.set("colorSpace", colorSpace)
    object.set("alphaMode", alphaMode)
    object.set("backgroundColor", backgroundColor)
    object.set("masterGain", masterGain)
    object.set("toneMapping", toneMapping)
    object.set("exposure", exposure)
    object.set("target", target)
    return object.json
  }
}

extension InterchangeAsset {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    id = try object.requiredString("id")
    kind = try object.requiredString("kind")
    src = try object.requiredString("src")
    name = object.string("name")
    duration = object.double("duration")
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("id", id)
    object.set("kind", kind)
    object.set("src", src)
    object.set("name", name)
    object.set("duration", duration)
    return object.json
  }
}

extension InterchangeSequence {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    id = try object.requiredString("id")
    name = object.string("name")
    duration = object.double("duration", default: 0)
    tracks = try decodeArray(object.array("tracks"), InterchangeTrack.init)
    transitions = try decodeArray(object.array("transitions"), InterchangeTransition.init)
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("id", id)
    object.set("name", name)
    object.set("duration", duration)
    object.set("tracks", jsonArray(tracks, \.json))
    object.set("transitions", jsonArray(transitions, \.json))
    return object.json
  }
}

extension InterchangeTrack {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    id = try object.requiredString("id")
    name = object.string("name")
    kind = try object.requiredString("kind")
    clips = try decodeArray(object.array("clips"), InterchangeClip.init)
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("id", id)
    object.set("name", name)
    object.set("kind", kind)
    object.set("clips", jsonArray(clips, \.json))
    return object.json
  }
}

extension InterchangeClip {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    id = try object.requiredString("id")
    timelineIn = try object.requiredDouble("timelineIn")
    timelineOut = try object.requiredDouble("timelineOut")
    sourceIn = object.double("sourceIn", default: 0)
    playbackRate = object.double("playbackRate", default: 1)
    compositionId = object.string("compositionId")
    assetId = object.string("assetId")
    enabled = object.bool("enabled")
    gain = object.double("gain")
    effects = try decodeArray(object.array("effects"), InterchangeEffect.init)
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("id", id)
    object.set("timelineIn", timelineIn)
    object.set("timelineOut", timelineOut)
    object.set("sourceIn", sourceIn)
    object.set("playbackRate", playbackRate)
    object.set("compositionId", compositionId)
    object.set("assetId", assetId)
    object.set("enabled", enabled)
    object.set("gain", gain)
    if !effects.isEmpty {
      object.set("effects", jsonArray(effects, \.json))
    }
    return object.json
  }
}

extension InterchangeTransition {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    id = try object.requiredString("id")
    fromClipId = try object.requiredString("fromClipId")
    toClipId = try object.requiredString("toClipId")
    kind = try object.requiredString("kind")
    duration = try object.requiredDouble("duration")
    easing = object.string("easing") ?? "linear"
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("id", id)
    object.set("fromClipId", fromClipId)
    object.set("toClipId", toClipId)
    object.set("kind", kind)
    object.set("duration", duration)
    object.set("easing", easing)
    return object.json
  }
}

extension InterchangeEffect {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    id = try object.requiredString("id")
    defId = try object.requiredString("defId")
    version = object.int("version", default: 1)
    enabled = object.bool("enabled", default: true)
    var decodedParams: [String: Double] = [:]
    if let params = try object.optionalObject("params") {
      for (key, value) in params.fields {
        if let number = value.number { decodedParams[key] = number }
      }
    }
    params = decodedParams
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("id", id)
    object.set("defId", defId)
    object.set("version", version)
    object.set("enabled", enabled)
    object.set(
      "params",
      .object(params.mapValues(JSONValue.double))
    )
    return object.json
  }
}

extension InterchangeComposition {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    id = try object.requiredString("id")
    name = object.string("name")
    duration = try object.requiredDouble("duration")
    spatial = try InterchangeSpatial(object.object("spatial").json)
    layers = try decodeArray(object.array("layers"), InterchangeLayer.init)
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("id", id)
    object.set("name", name)
    object.set("duration", duration)
    object.set("spatial", spatial.json)
    object.set("layers", jsonArray(layers, \.json))
    return object.json
  }
}

extension InterchangeSpatial {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    version = object.int("version", default: 1)
    programCamera = try InterchangeCamera(object.object("programCamera").json)
    environment =
      try object.optionalObject("environment").map { try InterchangeEnvironment($0.json) }
      ?? InterchangeEnvironment()
    lights = object.array("lights")
    exposure = object.double("exposure", default: 1)
    toneMapping = object.string("toneMapping") ?? "none"
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("version", version)
    object.set("programCamera", programCamera.json)
    object.set("environment", environment.json)
    object.set("lights", .array(lights))
    object.set("exposure", exposure)
    object.set("toneMapping", toneMapping)
    return object.json
  }
}

extension InterchangeCamera {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    kind = object.string("kind") ?? "orthographic"
    width = try object.requiredDouble("width")
    height = try object.requiredDouble("height")
    near = object.double("near", default: 0.1)
    far = object.double("far", default: 10_000)
    position = decodeVec(object.fields["position"], count: 3, default: [0, 0, 1_000])
    target = decodeVec(object.fields["target"], count: 3, default: [0, 0, 0])
    up = decodeVec(object.fields["up"], count: 3, default: [0, 1, 0])
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("kind", kind)
    object.set("width", width)
    object.set("height", height)
    object.set("near", near)
    object.set("far", far)
    object.set("position", jsonVec(position))
    object.set("target", jsonVec(target))
    object.set("up", jsonVec(up))
    return object.json
  }
}

extension InterchangeEnvironment {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    kind = object.string("kind") ?? "none"
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("kind", kind)
    return object.json
  }
}

extension InterchangeTransform {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    position = decodeVec(object.fields["position"], count: 3, default: [0, 0, 1])
    rotation = decodeVec(object.fields["rotation"], count: 3, default: [0, 0, 0])
    scale = decodeVec(object.fields["scale"], count: 3, default: [1, 1, 1])
    anchor = decodeVec(object.fields["anchor"], count: 3, default: [0, 0, 0])
  }

  var json: JSONValue {
    var object = JSONObject()
    object.set("position", jsonVec(position))
    object.set("rotation", jsonVec(rotation))
    object.set("scale", jsonVec(scale))
    object.set("anchor", jsonVec(anchor))
    return object.json
  }
}

extension InterchangeSurfaceTiming {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    timelineIn = try object.requiredDouble("timelineIn")
    timelineOut = try object.requiredDouble("timelineOut")
    sourceIn = object.double("sourceIn", default: 0)
    playbackRate = object.double("playbackRate", default: 1)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.set("timelineIn", timelineIn)
    object.set("timelineOut", timelineOut)
    object.set("sourceIn", sourceIn)
    object.set("playbackRate", playbackRate)
    return object.json
  }
}

extension InterchangeLayer {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    id = try object.requiredString("id")
    kind = try object.requiredString("kind")
    spatial = object.string("spatial") ?? "2d"
    transform =
      try object.optionalObject("transform").map { try InterchangeTransform($0.json) }
      ?? .identity
    opacity = object.double("opacity", default: 1)
    blendMode = object.string("blendMode") ?? "normal"
    visible = object.bool("visible", default: true)
    effects = try decodeArray(object.array("effects"), InterchangeEffect.init)
    assetId = object.string("assetId")
    timing = try object.optionalObject("timing").map {
      try InterchangeSurfaceTiming($0.json)
    }
    text = try object.optionalObject("text").map { try InterchangeTextContent($0.json) }
    canvasFit = object.string("canvasFit")
    canvasTranslationX = object.double("canvasTranslationX")
    canvasTranslationY = object.double("canvasTranslationY")
    canvasScale = object.double("canvasScale")
    canvasRotationRadians = object.double("canvasRotationRadians")
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("id", id)
    object.set("kind", kind)
    object.set("spatial", spatial)
    object.set("transform", transform.json)
    object.set("opacity", opacity)
    object.set("blendMode", blendMode)
    object.set("visible", visible)
    if !effects.isEmpty {
      object.set("effects", jsonArray(effects, \.json))
    }
    object.set("assetId", assetId)
    if let timing {
      object.set("timing", timing.json)
    }
    if let text {
      object.set("text", text.json)
    }
    object.set("canvasFit", canvasFit)
    object.set("canvasTranslationX", canvasTranslationX)
    object.set("canvasTranslationY", canvasTranslationY)
    object.set("canvasScale", canvasScale)
    object.set("canvasRotationRadians", canvasRotationRadians)
    return object.json
  }
}

extension InterchangeTextContent {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    self.value = try object.requiredString("value")
    engine = try InterchangeTextEngine(object.object("engine").json)
    spans = object.array("spans")
    timedWords = object.array("timedWords")
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("value", value)
    object.set("engine", engine.json)
    object.set("spans", .array(spans))
    object.set("timedWords", .array(timedWords))
    return object.json
  }
}

extension InterchangeTextEngine {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    version = object.int("version", default: 1)
    semanticRevision = object.string("semanticRevision") ?? "h3ddle-text-v1"
    layoutEngineRevision = object.string("layoutEngineRevision") ?? "coretext"
    renderContractRevision = object.string("renderContractRevision") ?? "h3ddle-flat-text-v1"
    face = try InterchangeFontFace(object.object("face").json)
    fallbackFaces = try decodeArray(object.array("fallbackFaces"), InterchangeFontFace.init)
    render = try InterchangeTextRender(object.object("render").json)
    layout = try InterchangeTextLayout(object.object("layout").json)
    animatorStack =
      object.fields["animatorStack"]
      ?? .object(["version": .int(1), "animators": .array([])])
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("version", version)
    object.set("semanticRevision", semanticRevision)
    object.set("layoutEngineRevision", layoutEngineRevision)
    object.set("renderContractRevision", renderContractRevision)
    object.set("face", face.json)
    object.set("fallbackFaces", jsonArray(fallbackFaces, \.json))
    object.set("render", render.json)
    object.set("layout", layout.json)
    object.set("animatorStack", animatorStack)
    return object.json
  }
}

extension InterchangeFontFace {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    version = object.int("version", default: 1)
    let source = try object.optionalObject("source")
    sourceKind = source?.string("kind") ?? "builtin"
    faceId = source?.string("faceId") ?? object.string("faceId") ?? ".AppleSystemUIFont"
    sourceRevision = object.string("sourceRevision") ?? faceId
    faceIndex = object.int("faceIndex", default: 0)
    weight = object.int("weight", default: 400)
    style = object.string("style") ?? "normal"
    stretch = object.int("stretch", default: 100)
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("version", version)
    object.set(
      "source",
      .object([
        "kind": .string(sourceKind),
        "faceId": .string(faceId),
      ])
    )
    object.set("sourceRevision", sourceRevision)
    object.set("faceIndex", faceIndex)
    object.set("weight", weight)
    object.set("style", style)
    object.set("stretch", stretch)
    object.set(
      "axes",
      .object(["wght": .double(Double(weight))])
    )
    return object.json
  }
}

extension InterchangeTextRender {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    kind = object.string("kind") ?? "flat"
    let style = try object.optionalObject("style")
    let fillPaint = try style?.optionalObject("fill")
    fill = decodeVec(fillPaint?.fields["color"], count: 4, default: [1, 1, 1, 1])
    let stroke = try style?.optionalObject("stroke")
    strokeWidth = stroke?.double("width") ?? 0
    let strokePaint = try stroke?.optionalObject("paint")
    strokeColor = decodeVec(strokePaint?.fields["color"], count: 4, default: [0, 0, 0, 1])
    let background = try style?.optionalObject("background")
    let backgroundPaint = try background?.optionalObject("paint")
    backgroundColor = decodeVec(
      backgroundPaint?.fields["color"],
      count: 4,
      default: [0, 0, 0, 0]
    )
    backgroundPadding = background?.double("padding") ?? 0
    backgroundRadius = background?.double("radius") ?? 0
    let shadow = try style?.optionalObject("shadow")
    shadowColor = decodeVec(shadow?.fields["color"], count: 4, default: [0, 0, 0, 0.5])
    shadowBlur = shadow?.double("blur") ?? 0
    shadowOffset = decodeVec(shadow?.fields["offset"], count: 2, default: [0, 0])
  }

  var json: JSONValue {
    func paint(_ color: [Double]) -> JSONValue {
      .object([
        "kind": .string("solid"),
        "color": jsonVec(color),
      ])
    }
    let hasStroke = strokeWidth > 0
    let hasBackground = (backgroundColor[safe: 3] ?? 0) > 0 || backgroundPadding > 0
    let hasShadow =
      shadowBlur > 0 || (shadowOffset[safe: 0] ?? 0) != 0 || (shadowOffset[safe: 1] ?? 0) != 0
      || (shadowColor[safe: 3] ?? 0) > 0
    var style = JSONObject()
    style.set("fill", paint(fill))
    style.set(
      "stroke",
      hasStroke
        ? .object(["paint": paint(strokeColor), "width": .double(strokeWidth)])
        : .null
    )
    style.set(
      "background",
      hasBackground
        ? .object([
          "paint": paint(backgroundColor),
          "padding": .double(backgroundPadding),
          "radius": .double(backgroundRadius),
        ])
        : .null
    )
    style.set(
      "shadow",
      hasShadow
        ? .object([
          "color": jsonVec(shadowColor),
          "blur": .double(shadowBlur),
          "offset": jsonVec(shadowOffset),
        ])
        : .null
    )
    return .object([
      "kind": .string(kind),
      "style": style.json,
    ])
  }
}

extension InterchangeTextLayout {
  init(_ value: JSONValue) throws {
    let object = try JSONObject(value)
    version = object.int("version", default: 1)
    fontSize = try object.requiredDouble("fontSize")
    lineHeight = object.double("lineHeight", default: 1.2)
    letterSpacing = object.double("letterSpacing", default: 0)
    language = object.string("language") ?? "und"
    direction = object.string("direction") ?? "auto"
    align = object.string("align") ?? "center"
    wrap = object.string("wrap") ?? "none"
    let box = try object.optionalObject("box")
    boxKind = box?.string("kind") ?? "auto"
    boxWidth = box?.double("width")
    extras = object.extras(excluding: Self.knownKeys)
  }

  var json: JSONValue {
    var object = JSONObject()
    object.mergeExtras(extras, known: Self.knownKeys)
    object.set("version", version)
    object.set("fontSize", fontSize)
    object.set("lineHeight", lineHeight)
    object.set("letterSpacing", letterSpacing)
    object.set("language", language)
    object.set("direction", direction)
    object.set("features", .object([:]))
    object.set("whiteSpace", "preserve")
    object.set("tabStops", .array([]))
    object.set("align", align)
    var box = JSONObject()
    box.set("kind", boxKind)
    if boxKind == "fixed" {
      box.set("width", boxWidth ?? 0)
      box.set("height", .null)
    }
    object.set("box", box.json)
    object.set("wrap", wrap)
    object.set("overflow", "visible")
    object.set("autoFit", "none")
    object.set("hyphenation", "none")
    return object.json
  }
}
