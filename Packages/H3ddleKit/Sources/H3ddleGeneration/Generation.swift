import Foundation
import H3ddleCore
import H3ddleEngineProtocol

public enum GenerationKind: String, CaseIterable, Codable, Identifiable, Sendable {
  case video
  case image
  case audio

  public var id: String { rawValue }

  public var mediaKind: MediaKind {
    switch self {
    case .video: .video
    case .image: .image
    case .audio: .audio
    }
  }
}

/// Which model an audio generation runs through. All three write a WAV, so
/// nothing downstream differs; what differs is the package and what it can be
/// asked for. H3 writes the joint soundtrack that accompanies its video,
/// Stable Audio makes music and sound effects, and Qwen3-TTS speaks a written
/// line in a voice cloned from a reference clip.
public enum AudioGenerationEngine: String, Codable, Sendable {
  case h3
  case stableAudio
  case speech

  /// Whether this engine loads a package of its own rather than the H3 tree.
  public var usesOwnPackage: Bool { self != .h3 }
}

/// Which model a still runs through. Both write a PNG, so nothing downstream
/// differs; what differs is the package and the quality. H3 makes a still by
/// rendering a very short clip and keeping a frame, which is why the video
/// knobs apply to it. Z-Image-Turbo is a dedicated text-to-image model.
public enum ImageGenerationEngine: String, Codable, Sendable {
  case h3
  case zImage

  /// Whether this engine loads a package of its own rather than the H3 tree.
  public var usesOwnPackage: Bool { self != .h3 }
}

/// One installed model the image lane can choose between. This small domain
/// value keeps the default-selection rule independent from SwiftUI and from
/// how the app discovered the model on disk.
public struct ImageGenerationModelOption: Sendable {
  public let id: String
  public let engine: ImageGenerationEngine
  public let isSelectedVideoModel: Bool

  public init(
    id: String,
    engine: ImageGenerationEngine,
    isSelectedVideoModel: Bool = false
  ) {
    self.id = id
    self.engine = engine
    self.isSelectedVideoModel = isSelectedVideoModel
  }
}

/// Resolves the image pick without manufacturing a second engine setting.
/// An explicit installed choice wins; otherwise Z-Image is the default when
/// present, followed by the selected video model and then any remaining H3
/// package that supports still generation.
public enum ImageGenerationModelSelection {
  public static func preferredID(
    among options: [ImageGenerationModelOption],
    selectedID: String?
  ) -> String? {
    if let selectedID, options.contains(where: { $0.id == selectedID }) {
      return selectedID
    }
    if let zImage = options.first(where: { $0.engine == .zImage }) {
      return zImage.id
    }
    return options.first(where: \.isSelectedVideoModel)?.id ?? options.first?.id
  }
}

/// Which engine renders a clip.
///
/// Both write video with a soundtrack, so nothing about the *output* tells
/// them apart — only the package does. H3 is the resident engine with the
/// reference inputs, keyframes, previews and quality ladder built around it;
/// LTX-2.5 renders markedly better motion in eight steps and supports none of
/// that.
public enum VideoGenerationEngine: String, Codable, Sendable {
  case h3
  case ltx

  /// Whether this engine loads a package of its own rather than the H3 tree.
  public var usesOwnPackage: Bool { self != .h3 }

  /// Whether this video package also implements the app's still-generation
  /// contract. H3 keeps a decoded frame from a short temporal generation;
  /// LTX currently has only its video/MP4 service path.
  public var supportsStillGeneration: Bool { self == .h3 }

  /// Whether a clip from this engine can be conditioned on pictures. Both can
  /// now, by different means: H3 reads them as keyframes and Ref2VA stills,
  /// LTX encodes them and appends their tokens to the DiT's own sequence.
  public var acceptsReferenceInputs: Bool { true }

  /// How many pictures this engine takes in total. LTX's ceiling is a cost
  /// rather than a capability: each one is a full VAE encode and a permanent
  /// addition to the sequence every block of every step reads.
  public var conditioningLimit: Int { self == .ltx ? 4 : 9 }
}

public struct GenerationRequest: Hashable, Codable, Sendable {
  public var kind: GenerationKind
  /// Audio only; ignored by the other kinds.
  public var audioEngine: AudioGenerationEngine = .h3
  /// Image only; ignored by the other kinds.
  public var imageEngine: ImageGenerationEngine = .h3
  /// Video only; ignored by the other kinds.
  public var videoEngine: VideoGenerationEngine = .h3
  /// H3 video only. When present, the source clip is partly regenerated
  /// through the Ref2VA transformer according to its hard mask.
  public var videoInpainting: EngineVideoInpaintingOptions?
  /// Required when `audioEngine` is `.speech`, ignored otherwise: the line to
  /// speak travels in `prompt`, and everything else about the voice here.
  public var speech: EngineSpeechOptions?
  public var prompt: String
  public var duration: TimeInterval
  public var quality: EngineGenerationQuality
  /// Overrides the preset's denoising budget when set; nil keeps the preset.
  public var denoisingSteps: Int?
  /// Overrides the preset's retained DiT blocks when set; nil keeps the preset.
  public var activeDiTLayers: Int?
  /// Transformer-core reuse interval; nil or 1 keeps the exact path.
  public var coreReuse: Int?
  /// Render stills from a 5-frame clip instead of 22: about 3x faster with
  /// visibly less detail. Image generations only.
  public var fastStill: Bool
  /// Replay cached tail-block residuals on schedule-gated steps: about 40%
  /// faster at standard step counts for a different sample of the same
  /// quality. Replaces both reuse ladders when enabled.
  public var blockCache: Bool
  /// Decode a still after every denoising pass. Off by default; does not
  /// change the encoded video.
  public var previewDenoise: Bool
  /// Beta(0.6, 0.6) sigma spacing; the schedule turbo checkpoints expect.
  public var useBetaSchedule: Bool
  /// Random-stream seed; nil keeps the engine default. Same seed and
  /// settings reproduce a generation.
  public var seed: UInt64?
  /// How much of a start picture a still model repaints, 0 through 1. Nil
  /// where the lane takes no picture, or where none was given.
  public var sourceStrength: Double?
  /// Overrides the quality preset's square canvas when both are set.
  public var canvasWidth: Int?
  public var canvasHeight: Int?
  public var firstFrameURL: URL?
  public var lastFrameURL: URL?
  public var referenceImageURLs: [URL]
  /// True only after the user accepts the high unified-memory warning for an
  /// LTX request. The engine independently enforces the same estimate.
  public var allowsLTXMemoryOvercommit: Bool?

  public init(
    kind: GenerationKind,
    audioEngine: AudioGenerationEngine = .h3,
    imageEngine: ImageGenerationEngine = .h3,
    videoEngine: VideoGenerationEngine = .h3,
    videoInpainting: EngineVideoInpaintingOptions? = nil,
    speech: EngineSpeechOptions? = nil,
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
    sourceStrength: Double? = nil,
    canvasWidth: Int? = nil,
    canvasHeight: Int? = nil,
    firstFrameURL: URL? = nil,
    lastFrameURL: URL? = nil,
    referenceImageURLs: [URL] = [],
    allowsLTXMemoryOvercommit: Bool = false
  ) {
    self.kind = kind
    self.audioEngine = audioEngine
    self.imageEngine = imageEngine
    self.videoEngine = videoEngine
    self.videoInpainting = videoInpainting
    self.speech = speech
    self.prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    self.duration = max(0, duration)
    self.quality = quality
    self.denoisingSteps = denoisingSteps
    self.activeDiTLayers = activeDiTLayers
    self.coreReuse = coreReuse
    self.fastStill = fastStill
    self.blockCache = blockCache
    self.previewDenoise = previewDenoise
    self.useBetaSchedule = useBetaSchedule
    self.seed = seed
    self.sourceStrength = sourceStrength
    self.canvasWidth = canvasWidth
    self.canvasHeight = canvasHeight
    self.firstFrameURL = firstFrameURL
    self.lastFrameURL = lastFrameURL
    self.referenceImageURLs = Array(
      referenceImageURLs.prefix(EngineGenerationRequest.referenceImageLimit))
    self.allowsLTXMemoryOvercommit = allowsLTXMemoryOvercommit
  }
}

public enum GenerationEvent: Hashable, Sendable {
  case progress(phase: String, fractionComplete: Double)
  case preview(URL)
  case completed(AssetReference)
}

public protocol GenerationProvider: Sendable {
  func events(for request: GenerationRequest) -> AsyncThrowingStream<GenerationEvent, any Error>
}

public enum GenerationError: LocalizedError, Equatable, Sendable {
  case emptyPrompt

  public var errorDescription: String? {
    switch self {
    case .emptyPrompt:
      "Describe what you want to generate."
    }
  }
}
