import Foundation

public enum H3ddleEngineProtocol {
  public static let currentVersion = 14
}

public enum EngineCommandKind: String, Codable, Sendable {
  case handshake
  case inspectModel
  case generate
  case cancel
  case shutdown
}

public enum EngineGenerationKind: String, Codable, Sendable {
  case video
  case image
  case audio
  /// Stable Audio 3 rather than H3: sound effects and ambience, which
  /// H3's dialogue-trained audio branch will not produce.
  case soundEffect
  /// Qwen3-TTS: a written line spoken in a voice cloned from a reference
  /// clip. H3's audio branch produces dialogue, but not chosen words in a
  /// chosen voice.
  case speech
}

/// The ten languages the speech model was released with. An enum rather than
/// a string because the engine refuses an unknown code outright: the wrong
/// language token still produces fluent speech, just in the wrong accent, so
/// a typo that fell back to English would be a bug nobody reports.
public enum EngineSpeechLanguage: String, CaseIterable, Codable, Sendable {
  case english = "en"
  case chinese = "zh"
  case german = "de"
  case spanish = "es"
  case french = "fr"
  case italian = "it"
  case portuguese = "pt"
  case russian = "ru"
  case japanese = "ja"
  case korean = "ko"

  public var displayName: String {
    switch self {
    case .english: "English"
    case .chinese: "Chinese"
    case .german: "German"
    case .spanish: "Spanish"
    case .french: "French"
    case .italian: "Italian"
    case .portuguese: "Portuguese"
    case .russian: "Russian"
    case .japanese: "Japanese"
    case .korean: "Korean"
    }
  }
}

/// Everything a speech job needs beyond the prompt, which carries the line to
/// speak, and the duration, which caps it.
public struct EngineSpeechOptions: Hashable, Codable, Sendable {
  /// Zero is greedy, which is reproducible and a bad default: greedy decoding
  /// loops. A six-word line measured at temperature 0 ran to a thirty-second
  /// ceiling where the same line at 0.7 or 0.9 stopped after 2.3 seconds.
  public static let temperatureRange = 0.0...2.0
  public static let topKRange = 0...200
  public static let repetitionPenaltyRange = 1.0...2.0
  public static let defaultTemperature = 0.9
  public static let defaultTopK = 50
  public static let defaultRepetitionPenalty = 1.05

  /// A clip to clone from: any audio file the system can decode. A few
  /// seconds of clean speech is both the minimum and about all it needs,
  /// since the encoder pools over the whole thing.
  public var referenceAudioURL: URL?
  /// A voice saved earlier — the 1024 numbers the encoder took from a clip,
  /// which are the whole of what the model conditions on. Takes precedence
  /// over a clip; with neither, the model speaks unconditioned.
  public var voiceEmbeddingURL: URL?
  public var language: EngineSpeechLanguage
  public var temperature: Double
  /// 0 for no restriction.
  public var topK: Int
  /// 1.0 for none.
  public var repetitionPenalty: Double

  public init(
    referenceAudioURL: URL? = nil,
    voiceEmbeddingURL: URL? = nil,
    language: EngineSpeechLanguage = .english,
    temperature: Double = EngineSpeechOptions.defaultTemperature,
    topK: Int = EngineSpeechOptions.defaultTopK,
    repetitionPenalty: Double = EngineSpeechOptions.defaultRepetitionPenalty
  ) {
    self.referenceAudioURL = referenceAudioURL
    self.voiceEmbeddingURL = voiceEmbeddingURL
    self.language = language
    self.temperature = Self.temperatureRange.clamping(temperature)
    self.topK = Self.topKRange.clamping(topK)
    self.repetitionPenalty = Self.repetitionPenaltyRange.clamping(repetitionPenalty)
  }
}

/// Which model renders a still.
///
/// H3 produces one by generating a very short clip and keeping a frame, which
/// is why the request's video knobs — retained DiT blocks, block cache, beta
/// schedule — apply to it and mean nothing to the other. Z-Image-Turbo is a
/// dedicated text-to-image model: markedly better pictures, eight forwards
/// rather than a clip's worth, and its own package.
public enum EngineImageModel: String, Codable, Sendable {
  case h3
  case zImage
}

/// Settings that apply to a still and to nothing else. Absent means H3, which
/// is what `.image` meant before this existed.
public struct EngineImageOptions: Hashable, Codable, Sendable {
  /// Z-Image-Turbo is step-distilled and released at eight; fewer trades
  /// detail for time and more buys very little.
  public static let stepsRange = 1...32

  public var model: EngineImageModel
  /// Overrides the model's released schedule when set. Ignored by H3, which
  /// takes its budget from `denoisingSteps`.
  public var steps: Int?

  public init(model: EngineImageModel = .h3, steps: Int? = nil) {
    self.model = model
    self.steps = steps.map { Self.stepsRange.clamping($0) }
  }
}

public enum EngineFeature: String, CaseIterable, Codable, Sendable {
  case modelInspection
  case videoGeneration
  case imageGeneration
  /// Z-Image-Turbo: a dedicated text-to-image model, as against the still H3
  /// makes by rendering a very short clip and keeping a frame. Separate from
  /// `imageGeneration` because an engine can have one package and not the
  /// other, and the app must not offer a model the engine cannot load.
  case zImageGeneration
  case standaloneAudioGeneration
  case soundEffectGeneration
  case speechGeneration
  case embeddedAudio
  case cancellation
  case denoisingPreviews
  case referenceInputs
}

public struct EngineCapabilities: Hashable, Codable, Sendable {
  public var engineName: String
  public var engineVersion: String
  public var features: [EngineFeature]

  public init(
    engineName: String,
    engineVersion: String,
    features: [EngineFeature]
  ) {
    self.engineName = engineName
    self.engineVersion = engineVersion
    self.features = EngineFeature.allCases.filter { features.contains($0) }
  }

  public func supports(_ feature: EngineFeature) -> Bool {
    features.contains(feature)
  }
}

public struct EngineModelInspectionRequest: Hashable, Codable, Sendable {
  public var modelDirectory: URL

  public init(modelDirectory: URL) {
    self.modelDirectory = modelDirectory
  }
}

public enum EngineModelComponentKind: String, CaseIterable, Codable, Sendable {
  case textEncoder
  case videoTransformer
  case referenceTransformer
  case videoVAE
  case audioVAE
}

public enum EngineModelFormat: String, Codable, Sendable {
  case unknown
  case releasedDirectory
  case optimizedINT8SingleFile
}

public struct EngineModelComponent: Hashable, Codable, Sendable {
  public var kind: EngineModelComponentKind
  public var bytes: UInt64
  public var tensorBytes: UInt64
  public var fileCount: Int
  public var tensorCount: Int

  public init(
    kind: EngineModelComponentKind,
    bytes: UInt64,
    tensorBytes: UInt64,
    fileCount: Int,
    tensorCount: Int
  ) {
    self.kind = kind
    self.bytes = bytes
    self.tensorBytes = tensorBytes
    self.fileCount = max(0, fileCount)
    self.tensorCount = max(0, tensorCount)
  }
}

public struct EngineDeviceReport: Hashable, Codable, Sendable {
  public var name: String
  public var architecture: String
  public var physicalMemory: UInt64
  public var recommendedWorkingSet: UInt64
  public var unifiedMemory: Bool

  public init(
    name: String,
    architecture: String,
    physicalMemory: UInt64,
    recommendedWorkingSet: UInt64,
    unifiedMemory: Bool
  ) {
    self.name = name
    self.architecture = architecture
    self.physicalMemory = physicalMemory
    self.recommendedWorkingSet = recommendedWorkingSet
    self.unifiedMemory = unifiedMemory
  }
}

public struct EngineModelReport: Hashable, Codable, Sendable {
  public var modelDirectory: URL
  public var components: [EngineModelComponent]
  public var device: EngineDeviceReport
  public var format: EngineModelFormat
  public var supportsGeneration: Bool

  public init(
    modelDirectory: URL,
    components: [EngineModelComponent],
    device: EngineDeviceReport,
    format: EngineModelFormat = .releasedDirectory,
    supportsGeneration: Bool = true
  ) {
    self.modelDirectory = modelDirectory
    self.components = EngineModelComponentKind.allCases.compactMap { kind in
      components.first { $0.kind == kind }
    }
    self.device = device
    self.format = format
    self.supportsGeneration = supportsGeneration
  }

  public var totalBytes: UInt64 {
    components.reduce(0) { $0 + $1.bytes }
  }

  public var hasReferenceTransformer: Bool {
    components.first { $0.kind == .referenceTransformer }?.fileCount ?? 0 > 0
  }
}

/// Validated speed/quality presets for H3 generation. Each tier maps to a
/// combination the vendored h3.c documentation has validated end to end;
/// arbitrary parameter mixes are deliberately not exposed here.
public enum EngineGenerationQuality: String, CaseIterable, Codable, Sendable {
  /// Fastest recognizable output: the engine's native preview canvas with its
  /// minimum validated denoising budget. Sized for M1/M2-class development.
  case preview
  /// The repeatedly validated development canvas with the documented fast
  /// settings (thinned blocks, denoiser reuse). Sized for M3/M4-class Macs.
  case standard
  /// Close-quality square output with every block and evaluation fresh.
  /// Sized for M5-class Macs; slow on anything earlier.
  case high

  /// Output canvas edge. All presets use square canvases; H3 requires
  /// multiples of 32. 448 square is 0.2 megapixels, the smallest canvas the
  /// reference workflows validate — below it prompts stop steering the scene
  /// and outputs become functions of the seed, measured 2026-08-14 across
  /// prompts, formats, models, and step counts at 256 square.
  public var canvasSize: Int {
    switch self {
    case .preview: 448
    case .standard: 512
    case .high: 768
    }
  }

  /// Denoising passes. Four is the minimum validated budget; twenty is the
  /// engine default schedule.
  public var denoisingSteps: Int {
    switch self {
    case .preview: 4
    case .standard, .high: 20
    }
  }

  /// Gate-ranked DiT residual blocks to retain. 50 is exact; 45 is the
  /// validated fast setting.
  public var activeDiTLayers: Int {
    switch self {
    case .preview, .high: 50
    case .standard: 45
    }
  }

  /// Whole-denoiser reuse interval. Must stay 1 at small step budgets so
  /// every requested pass runs the model; 2 is the validated fast path.
  public var denoiseReuse: Int {
    switch self {
    case .preview, .high: 1
    case .standard: 2
    }
  }
}

public struct EngineGenerationRequest: Hashable, Codable, Sendable {
  /// H3 accepts denoising budgets in [2, 1000]; the app UI exposes a much
  /// narrower band, but the protocol clamps to the engine's true range.
  public static let denoisingStepsRange = 2...1000
  /// Gate-ranked DiT residual blocks to retain; the engine rejects
  /// anything outside [35, 50].
  public static let activeDiTLayersRange = 35...50
  /// Transformer-core reuse interval; the engine rejects anything
  /// outside [1, 6]. Values above 1 disable whole-denoiser reuse.
  public static let coreReuseRange = 1...6

  public var kind: EngineGenerationKind
  public var prompt: String
  public var duration: TimeInterval
  public var quality: EngineGenerationQuality
  /// Overrides the preset's denoising budget when set; nil keeps the preset.
  public var denoisingSteps: Int?
  /// Overrides the preset's retained DiT blocks when set; nil keeps the preset.
  public var activeDiTLayers: Int?
  /// Recomputes the expensive transformer core every N denoising passes while
  /// refreshing the timestep heads each pass; nil or 1 keeps the exact path.
  public var coreReuse: Int?
  /// Render stills from the 5-frame first chunk instead of a 22-frame clip:
  /// about 3x faster, with visibly less detail. Ignored by video and audio.
  public var fastStill: Bool
  /// Replay cached tail-block residuals on schedule-gated denoising steps:
  /// about 40% faster at standard step counts, trading exact reproduction for
  /// a different sample of the same quality. Replaces both reuse ladders.
  public var blockCache: Bool
  /// Decode a representative still after every Euler step. Off by default
  /// because each preview is a full VideoVAE pass and does not change the
  /// final MP4.
  public var previewDenoise: Bool
  /// Space sigmas at Beta(0.6, 0.6) quantiles instead of the released linear
  /// grid. Step-distilled turbo checkpoints are trained against this spacing.
  public var useBetaSchedule: Bool
  /// Random-stream seed for the native generators; nil keeps the engine
  /// default (42). Identical seed + settings reproduce a generation.
  public var seed: UInt64?
  /// Overrides the quality preset's square canvas when both are set.
  public var canvasWidth: Int?
  public var canvasHeight: Int?
  /// FL2VA keyframe anchors. Mutually exclusive with `referenceImageURLs`.
  public var firstFrameURL: URL?
  public var lastFrameURL: URL?
  /// Ordered Ref2VA stills. Mutually exclusive with first/last frames.
  public var referenceImageURLs: [URL]
  /// Ref2VA takes 12 references in total but no more than 9 images, and every
  /// reference sent from here is an image, so 9 is the ceiling that applies.
  public static let referenceImageLimit = 9
  /// Square canvas used for audio jobs: the mechanical minimum, since the
  /// pictures are discarded. Raising it does not improve prompt adherence in
  /// the audio lane — 32 through 256 were compared and all returned speech.
  public static let audioCanvasSize = 32
  /// Required by `.speech` and ignored by every other kind.
  public var speech: EngineSpeechOptions?
  /// Which model renders a still, and its own settings. Absent keeps H3.
  public var image: EngineImageOptions?
  public var modelDirectory: URL?
  public var outputURL: URL

  public init(
    kind: EngineGenerationKind,
    prompt: String,
    duration: TimeInterval,
    quality: EngineGenerationQuality = .preview,
    denoisingSteps: Int? = nil,
    activeDiTLayers: Int? = nil,
    coreReuse: Int? = nil,
    fastStill: Bool = false,
    blockCache: Bool = false,
    previewDenoise: Bool = false,
    useBetaSchedule: Bool = false,
    seed: UInt64? = nil,
    canvasWidth: Int? = nil,
    canvasHeight: Int? = nil,
    firstFrameURL: URL? = nil,
    lastFrameURL: URL? = nil,
    referenceImageURLs: [URL] = [],
    speech: EngineSpeechOptions? = nil,
    image: EngineImageOptions? = nil,
    modelDirectory: URL? = nil,
    outputURL: URL
  ) {
    self.kind = kind
    self.prompt = prompt
    self.duration = max(0, duration)
    self.quality = quality
    self.denoisingSteps = denoisingSteps.map { Self.denoisingStepsRange.clamping($0) }
    self.activeDiTLayers = activeDiTLayers.map { Self.activeDiTLayersRange.clamping($0) }
    self.coreReuse = coreReuse.map { Self.coreReuseRange.clamping($0) }
    self.fastStill = fastStill
    self.blockCache = blockCache
    self.previewDenoise = previewDenoise
    self.useBetaSchedule = useBetaSchedule
    self.seed = seed
    self.canvasWidth = canvasWidth
    self.canvasHeight = canvasHeight
    self.firstFrameURL = firstFrameURL
    self.lastFrameURL = lastFrameURL
    self.referenceImageURLs = Array(referenceImageURLs.prefix(Self.referenceImageLimit))
    self.speech = speech
    self.image = image
    self.modelDirectory = modelDirectory
    self.outputURL = outputURL
  }

  enum CodingKeys: String, CodingKey {
    case kind
    case prompt
    case duration
    case quality
    case denoisingSteps
    case activeDiTLayers
    case coreReuse
    case fastStill
    case blockCache
    case previewDenoise
    case useBetaSchedule
    case seed
    case canvasWidth
    case canvasHeight
    case firstFrameURL
    case lastFrameURL
    case referenceImageURLs
    case speech
    case image
    case modelDirectory
    case outputURL
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    kind = try container.decode(EngineGenerationKind.self, forKey: .kind)
    prompt = try container.decode(String.self, forKey: .prompt)
    duration = max(0, try container.decode(TimeInterval.self, forKey: .duration))
    quality = try container.decodeIfPresent(EngineGenerationQuality.self, forKey: .quality)
      ?? .preview
    denoisingSteps = try container.decodeIfPresent(Int.self, forKey: .denoisingSteps)
      .map { Self.denoisingStepsRange.clamping($0) }
    activeDiTLayers = try container.decodeIfPresent(Int.self, forKey: .activeDiTLayers)
      .map { Self.activeDiTLayersRange.clamping($0) }
    coreReuse = try container.decodeIfPresent(Int.self, forKey: .coreReuse)
      .map { Self.coreReuseRange.clamping($0) }
    fastStill = try container.decodeIfPresent(Bool.self, forKey: .fastStill) ?? false
    blockCache = try container.decodeIfPresent(Bool.self, forKey: .blockCache) ?? false
    previewDenoise = try container.decodeIfPresent(Bool.self, forKey: .previewDenoise) ?? false
    useBetaSchedule =
      try container.decodeIfPresent(Bool.self, forKey: .useBetaSchedule) ?? false
    seed = try container.decodeIfPresent(UInt64.self, forKey: .seed)
    canvasWidth = try container.decodeIfPresent(Int.self, forKey: .canvasWidth)
    canvasHeight = try container.decodeIfPresent(Int.self, forKey: .canvasHeight)
    firstFrameURL = try container.decodeIfPresent(URL.self, forKey: .firstFrameURL)
    lastFrameURL = try container.decodeIfPresent(URL.self, forKey: .lastFrameURL)
    referenceImageURLs =
      try container.decodeIfPresent([URL].self, forKey: .referenceImageURLs) ?? []
    speech = try container.decodeIfPresent(EngineSpeechOptions.self, forKey: .speech)
    image = try container.decodeIfPresent(EngineImageOptions.self, forKey: .image)
    modelDirectory = try container.decodeIfPresent(URL.self, forKey: .modelDirectory)
    outputURL = try container.decode(URL.self, forKey: .outputURL)
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(kind, forKey: .kind)
    try container.encode(prompt, forKey: .prompt)
    try container.encode(duration, forKey: .duration)
    try container.encode(quality, forKey: .quality)
    try container.encodeIfPresent(denoisingSteps, forKey: .denoisingSteps)
    try container.encodeIfPresent(activeDiTLayers, forKey: .activeDiTLayers)
    try container.encodeIfPresent(coreReuse, forKey: .coreReuse)
    try container.encode(fastStill, forKey: .fastStill)
    try container.encode(blockCache, forKey: .blockCache)
    try container.encode(previewDenoise, forKey: .previewDenoise)
    try container.encode(useBetaSchedule, forKey: .useBetaSchedule)
    try container.encodeIfPresent(seed, forKey: .seed)
    try container.encodeIfPresent(canvasWidth, forKey: .canvasWidth)
    try container.encodeIfPresent(canvasHeight, forKey: .canvasHeight)
    try container.encodeIfPresent(firstFrameURL, forKey: .firstFrameURL)
    try container.encodeIfPresent(lastFrameURL, forKey: .lastFrameURL)
    if !referenceImageURLs.isEmpty {
      try container.encode(referenceImageURLs, forKey: .referenceImageURLs)
    }
    try container.encodeIfPresent(speech, forKey: .speech)
    try container.encodeIfPresent(image, forKey: .image)
    try container.encodeIfPresent(modelDirectory, forKey: .modelDirectory)
    try container.encode(outputURL, forKey: .outputURL)
  }
}

extension ClosedRange {
  fileprivate func clamping(_ value: Bound) -> Bound {
    Swift.min(Swift.max(value, lowerBound), upperBound)
  }
}

public struct EngineCommand: Hashable, Codable, Sendable {
  public var protocolVersion: Int
  public var requestID: UUID
  public var jobID: UUID?
  public var kind: EngineCommandKind
  public var modelInspection: EngineModelInspectionRequest?
  public var generation: EngineGenerationRequest?

  public init(
    protocolVersion: Int = H3ddleEngineProtocol.currentVersion,
    requestID: UUID = UUID(),
    jobID: UUID? = nil,
    kind: EngineCommandKind,
    modelInspection: EngineModelInspectionRequest? = nil,
    generation: EngineGenerationRequest? = nil
  ) {
    self.protocolVersion = protocolVersion
    self.requestID = requestID
    self.jobID = jobID
    self.kind = kind
    self.modelInspection = modelInspection
    self.generation = generation
  }
}

public enum EngineEventKind: String, Codable, Sendable {
  case ready
  case modelInspected
  case accepted
  case progress
  case preview
  case completed
  case cancelled
  case failed
  /// Weights entered or left memory. The helper evicts on idle and under
  /// memory pressure, so this cannot be inferred from job activity: it is
  /// reported when it happens or the app's picture goes stale.
  case residency
}

/// Which weights the helper is currently holding.
///
/// Only the video model is tracked. It is the one worth reporting: at tens
/// of gigabytes, whether it is resident decides whether the next generation
/// starts immediately or reloads for minutes. Audio packages are small
/// enough that their residency is not a decision the user makes.
public struct EngineResidency: Hashable, Codable, Sendable {
  /// Directory of the loaded video package, or nil when none is resident.
  public var videoModelDirectory: URL?

  public init(videoModelDirectory: URL? = nil) {
    self.videoModelDirectory = videoModelDirectory
  }
}

public struct EngineEvent: Hashable, Codable, Sendable {
  public var protocolVersion: Int
  public var requestID: UUID
  public var jobID: UUID?
  public var kind: EngineEventKind
  public var capabilities: EngineCapabilities?
  public var model: EngineModelReport?
  public var residency: EngineResidency?
  public var phase: String?
  public var fractionComplete: Double?
  public var outputURL: URL?
  public var outputDuration: TimeInterval?
  public var message: String?

  public init(
    protocolVersion: Int = H3ddleEngineProtocol.currentVersion,
    requestID: UUID,
    jobID: UUID? = nil,
    kind: EngineEventKind,
    capabilities: EngineCapabilities? = nil,
    model: EngineModelReport? = nil,
    residency: EngineResidency? = nil,
    phase: String? = nil,
    fractionComplete: Double? = nil,
    outputURL: URL? = nil,
    outputDuration: TimeInterval? = nil,
    message: String? = nil
  ) {
    self.protocolVersion = protocolVersion
    self.requestID = requestID
    self.jobID = jobID
    self.kind = kind
    self.capabilities = capabilities
    self.model = model
    self.residency = residency
    self.phase = phase
    self.fractionComplete = fractionComplete.map { min(max($0, 0), 1) }
    self.outputURL = outputURL
    self.outputDuration = outputDuration.map { max(0, $0) }
    self.message = message
  }
}

public enum EngineLineCodec {
  private static let encoder: JSONEncoder = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return encoder
  }()

  private static let decoder = JSONDecoder()

  public static func encode<T: Encodable>(_ value: T) throws -> Data {
    var data = try encoder.encode(value)
    data.append(0x0A)
    return data
  }

  public static func decode<T: Decodable>(_ type: T.Type, from line: Data) throws -> T {
    try decoder.decode(type, from: line)
  }
}
