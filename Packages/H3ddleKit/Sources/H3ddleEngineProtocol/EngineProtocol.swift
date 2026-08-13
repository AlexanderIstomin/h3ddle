import Foundation

public enum H3ddleEngineProtocol {
  public static let currentVersion = 11
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
}

public enum EngineFeature: String, CaseIterable, Codable, Sendable {
  case modelInspection
  case videoGeneration
  case imageGeneration
  case standaloneAudioGeneration
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
  /// multiples of 32 and treats 256 as the smallest recognizable size.
  public var canvasSize: Int {
    switch self {
    case .preview: 256
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
  /// Adapter contribution accepted by the engine.
  public static let adapterStrengthRange = -2.0...2.0

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
  /// Low-rank adapter applied to the transformer at runtime, so a
  /// distillation can ship as a small file instead of a merged checkpoint.
  public var adapterURL: URL?
  /// Adapter contribution; the engine accepts [-2, 2] and 1 is the
  /// strength the published adapters are trained for.
  public var adapterStrength: Double?
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
    previewDenoise: Bool = false,
    useBetaSchedule: Bool = false,
    seed: UInt64? = nil,
    adapterURL: URL? = nil,
    adapterStrength: Double? = nil,
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
    self.previewDenoise = previewDenoise
    self.useBetaSchedule = useBetaSchedule
    self.seed = seed
    self.adapterURL = adapterURL
    self.adapterStrength = adapterStrength.map {
      min(max($0, Self.adapterStrengthRange.lowerBound),
          Self.adapterStrengthRange.upperBound)
    }
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
    case previewDenoise
    case useBetaSchedule
    case seed
    case adapterURL
    case adapterStrength
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
    previewDenoise = try container.decodeIfPresent(Bool.self, forKey: .previewDenoise) ?? false
    useBetaSchedule =
      try container.decodeIfPresent(Bool.self, forKey: .useBetaSchedule) ?? false
    seed = try container.decodeIfPresent(UInt64.self, forKey: .seed)
    adapterURL = try container.decodeIfPresent(URL.self, forKey: .adapterURL)
    adapterStrength =
      try container.decodeIfPresent(Double.self, forKey: .adapterStrength).map {
        min(max($0, Self.adapterStrengthRange.lowerBound),
            Self.adapterStrengthRange.upperBound)
      }
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
    try container.encode(previewDenoise, forKey: .previewDenoise)
    try container.encode(useBetaSchedule, forKey: .useBetaSchedule)
    try container.encodeIfPresent(seed, forKey: .seed)
    try container.encodeIfPresent(adapterURL, forKey: .adapterURL)
    try container.encodeIfPresent(adapterStrength, forKey: .adapterStrength)
    try container.encodeIfPresent(modelDirectory, forKey: .modelDirectory)
    try container.encode(outputURL, forKey: .outputURL)
  }
}

extension ClosedRange<Int> {
  fileprivate func clamping(_ value: Int) -> Int {
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
}

public struct EngineEvent: Hashable, Codable, Sendable {
  public var protocolVersion: Int
  public var requestID: UUID
  public var jobID: UUID?
  public var kind: EngineEventKind
  public var capabilities: EngineCapabilities?
  public var model: EngineModelReport?
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
