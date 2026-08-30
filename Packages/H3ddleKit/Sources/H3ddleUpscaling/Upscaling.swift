import Foundation
import H3ddleCore

public struct UpscalingPixelSize: Hashable, Codable, Sendable {
  public var width: Int
  public var height: Int

  public init(width: Int, height: Int) {
    self.width = width
    self.height = height
  }

  public var isValid: Bool { width > 0 && height > 0 }
}

public enum UpscalingMode: String, CaseIterable, Codable, Identifiable, Sendable {
  /// Select the strongest temporally aware native backend available.
  case best
  /// Prefer a restoration model that may reconstruct detail absent from the source.
  case detailed
  /// Prefer a low-latency spatial scaler without a model download.
  case fast

  public var id: String { rawValue }
}

public enum UpscalingBackendID: String, CaseIterable, Codable, Sendable {
  case appleVideoToolbox = "apple-video-toolbox"
  case appleVideoToolboxLowLatency = "apple-video-toolbox-low-latency"
  case avFoundation = "av-foundation"
  case coreMLRealESRGAN = "coreml-real-esrgan"
  case accelerateVImage = "accelerate-vimage"
  case metalFX = "metal-fx"
  case fake
}

public enum UpscalingAvailability: Hashable, Codable, Sendable {
  case ready
  case modelDownloadRequired
  case unavailable(reason: String)
}

public struct UpscalingCapability: Hashable, Codable, Sendable {
  public var backend: UpscalingBackendID
  public var displayName: String
  public var supportedMediaKinds: [MediaKind]
  public var supportedScaleFactors: [Double]
  public var supportsArbitraryOutputSize: Bool
  public var usesTemporalFrames: Bool
  public var availability: UpscalingAvailability

  public init(
    backend: UpscalingBackendID,
    displayName: String,
    supportedMediaKinds: [MediaKind],
    supportedScaleFactors: [Double] = [],
    supportsArbitraryOutputSize: Bool,
    usesTemporalFrames: Bool,
    availability: UpscalingAvailability
  ) {
    self.backend = backend
    self.displayName = displayName
    self.supportedMediaKinds = supportedMediaKinds
    self.supportedScaleFactors = supportedScaleFactors
    self.supportsArbitraryOutputSize = supportsArbitraryOutputSize
    self.usesTemporalFrames = usesTemporalFrames
    self.availability = availability
  }
}

public struct UpscalingRequest: Hashable, Codable, Sendable {
  public var id: UUID
  public var sourceURL: URL
  public var sourceKind: MediaKind
  public var sourcePixelSize: UpscalingPixelSize
  public var sourceDuration: TimeInterval
  public var targetPixelSize: UpscalingPixelSize
  public var destinationURL: URL
  public var mode: UpscalingMode
  public var preservesAudio: Bool

  public init(
    id: UUID = UUID(),
    sourceURL: URL,
    sourceKind: MediaKind,
    sourcePixelSize: UpscalingPixelSize,
    sourceDuration: TimeInterval,
    targetPixelSize: UpscalingPixelSize,
    destinationURL: URL,
    mode: UpscalingMode = .best,
    preservesAudio: Bool = true
  ) {
    self.id = id
    self.sourceURL = sourceURL
    self.sourceKind = sourceKind
    self.sourcePixelSize = sourcePixelSize
    self.sourceDuration = max(0, sourceDuration)
    self.targetPixelSize = targetPixelSize
    self.destinationURL = destinationURL
    self.mode = mode
    self.preservesAudio = preservesAudio
  }

  public func validate() throws {
    guard sourceKind == .image || sourceKind == .video else {
      throw UpscalingError.unsupportedMediaKind
    }
    guard sourceURL.isFileURL, destinationURL.isFileURL else {
      throw UpscalingError.invalidFileURL
    }
    guard sourcePixelSize.isValid, targetPixelSize.isValid else {
      throw UpscalingError.invalidDimensions
    }
    guard
      targetPixelSize.width >= sourcePixelSize.width,
      targetPixelSize.height >= sourcePixelSize.height,
      targetPixelSize != sourcePixelSize
    else {
      throw UpscalingError.targetIsNotLarger
    }
    guard sourceURL.standardizedFileURL != destinationURL.standardizedFileURL else {
      throw UpscalingError.sourceAndDestinationMatch
    }
  }
}

public struct UpscalingResult: Hashable, Codable, Sendable {
  public var requestID: UUID
  public var outputURL: URL
  public var mediaKind: MediaKind
  public var pixelSize: UpscalingPixelSize
  public var duration: TimeInterval

  public init(
    requestID: UUID,
    outputURL: URL,
    mediaKind: MediaKind,
    pixelSize: UpscalingPixelSize,
    duration: TimeInterval
  ) {
    self.requestID = requestID
    self.outputURL = outputURL
    self.mediaKind = mediaKind
    self.pixelSize = pixelSize
    self.duration = max(0, duration)
  }
}

public enum UpscalingEvent: Hashable, Sendable {
  case preparing(backend: UpscalingBackendID)
  case progress(phase: String, fractionComplete: Double)
  case completed(UpscalingResult)
}

public protocol UpscalingProvider: Sendable {
  func capabilities(for request: UpscalingRequest) async -> [UpscalingCapability]
  func events(
    for request: UpscalingRequest
  ) -> AsyncThrowingStream<UpscalingEvent, any Error>
}

public enum UpscalingError: LocalizedError, Equatable, Sendable {
  case unsupportedMediaKind
  case invalidFileURL
  case invalidDimensions
  case targetIsNotLarger
  case sourceAndDestinationMatch
  case sourceNotReadable
  case destinationAlreadyExists
  case sourceDimensionsMismatch(expected: UpscalingPixelSize, actual: UpscalingPixelSize)
  case modelDownloadRequired
  case unsupportedScaleFactor
  case unavailable(String)
  case failed(String)

  public var errorDescription: String? {
    switch self {
    case .unsupportedMediaKind:
      "Only image and video assets can be upscaled."
    case .invalidFileURL:
      "Upscaling only supports local source and destination files."
    case .invalidDimensions:
      "The source and target dimensions must be positive."
    case .targetIsNotLarger:
      "The target must be larger than the source without shrinking either edge."
    case .sourceAndDestinationMatch:
      "Upscaling must create a new asset instead of overwriting its source."
    case .sourceNotReadable:
      "The source image could not be read."
    case .destinationAlreadyExists:
      "The upscale destination already exists."
    case .sourceDimensionsMismatch(let expected, let actual):
      "The source is \(actual.width)×\(actual.height), not the expected \(expected.width)×\(expected.height)."
    case .modelDownloadRequired:
      "Apple's high-quality upscaling model must be downloaded before using Detailed mode."
    case .unsupportedScaleFactor:
      "The requested output is larger than this upscaling backend supports."
    case .unavailable(let reason), .failed(let reason):
      reason
    }
  }
}
