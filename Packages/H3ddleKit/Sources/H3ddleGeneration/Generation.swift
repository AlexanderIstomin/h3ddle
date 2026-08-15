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

public struct GenerationRequest: Hashable, Codable, Sendable {
  public var kind: GenerationKind
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
  /// Overrides the quality preset's square canvas when both are set.
  public var canvasWidth: Int?
  public var canvasHeight: Int?
  public var firstFrameURL: URL?
  public var lastFrameURL: URL?
  public var referenceImageURLs: [URL]

  public init(
    kind: GenerationKind,
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
    referenceImageURLs: [URL] = []
  ) {
    self.kind = kind
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
    self.canvasWidth = canvasWidth
    self.canvasHeight = canvasHeight
    self.firstFrameURL = firstFrameURL
    self.lastFrameURL = lastFrameURL
    self.referenceImageURLs = Array(
      referenceImageURLs.prefix(EngineGenerationRequest.referenceImageLimit))
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
