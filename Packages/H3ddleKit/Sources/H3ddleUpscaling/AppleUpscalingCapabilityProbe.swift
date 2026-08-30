import Foundation
import H3ddleCore
import Metal
import MetalFX
import VideoToolbox

public enum UpscalingModelStatus: String, Codable, Sendable {
  case notRequired
  case downloadRequired
  case downloading
  case ready
}

/// Read-only snapshot of the upscalers exposed by the current Apple GPU and OS.
/// Inspecting this value never starts VideoToolbox's optional model download.
public struct AppleUpscalingCapabilitySnapshot: Hashable, Codable, Sendable {
  public var sourceKind: MediaKind
  public var sourcePixelSize: UpscalingPixelSize
  public var highQualitySupported: Bool
  public var highQualityScaleFactors: [Double]
  public var highQualityModelStatus: UpscalingModelStatus?
  public var lowLatencySupported: Bool
  public var lowLatencyScaleFactors: [Double]
  public var metalFXSupported: Bool

  public init(
    sourceKind: MediaKind,
    sourcePixelSize: UpscalingPixelSize,
    highQualitySupported: Bool,
    highQualityScaleFactors: [Double],
    highQualityModelStatus: UpscalingModelStatus?,
    lowLatencySupported: Bool,
    lowLatencyScaleFactors: [Double],
    metalFXSupported: Bool
  ) {
    self.sourceKind = sourceKind
    self.sourcePixelSize = sourcePixelSize
    self.highQualitySupported = highQualitySupported
    self.highQualityScaleFactors = highQualityScaleFactors
    self.highQualityModelStatus = highQualityModelStatus
    self.lowLatencySupported = lowLatencySupported
    self.lowLatencyScaleFactors = lowLatencyScaleFactors
    self.metalFXSupported = metalFXSupported
  }
}

public enum AppleUpscalingCapabilityProbe {
  public static func inspect(
    sourceKind: MediaKind,
    sourcePixelSize: UpscalingPixelSize
  ) -> AppleUpscalingCapabilitySnapshot {
    guard sourcePixelSize.isValid, sourceKind == .image || sourceKind == .video else {
      return AppleUpscalingCapabilitySnapshot(
        sourceKind: sourceKind,
        sourcePixelSize: sourcePixelSize,
        highQualitySupported: false,
        highQualityScaleFactors: [],
        highQualityModelStatus: nil,
        lowLatencySupported: false,
        lowLatencyScaleFactors: [],
        metalFXSupported: false
      )
    }

    let metalFXSupported = MTLCreateSystemDefaultDevice().map {
      MTLFXSpatialScalerDescriptor.supportsDevice($0)
    } ?? false

    guard #available(macOS 26.0, *) else {
      return AppleUpscalingCapabilitySnapshot(
        sourceKind: sourceKind,
        sourcePixelSize: sourcePixelSize,
        highQualitySupported: false,
        highQualityScaleFactors: [],
        highQualityModelStatus: nil,
        lowLatencySupported: false,
        lowLatencyScaleFactors: [],
        metalFXSupported: metalFXSupported
      )
    }

    let highQualitySupported = VTSuperResolutionScalerConfiguration.isSupported
    let highQualityScaleFactors = highQualitySupported
      ? VTSuperResolutionScalerConfiguration.supportedScaleFactors.map(Double.init)
      : []
    let inputType: VTSuperResolutionScalerConfiguration.InputType =
      sourceKind == .video ? .video : .image
    let dimensionCandidates = configurationDimensionCandidates(
      sourceKind: sourceKind,
      sourcePixelSize: sourcePixelSize
    )
    let firstConfiguration = highQualityScaleFactors.lazy.compactMap { factor in
      dimensionCandidates.lazy.compactMap { size in
        VTSuperResolutionScalerConfiguration(
          frameWidth: size.width,
          frameHeight: size.height,
          scaleFactor: Int(factor),
          inputType: inputType,
          usePrecomputedFlow: false,
          qualityPrioritization: .normal,
          revision: VTSuperResolutionScalerConfiguration.defaultRevision
        )
      }.first
    }.first
    let highQualityModelStatus = firstConfiguration.map { modelStatus($0.configurationModelStatus) }

    let lowLatencySupported = VTLowLatencySuperResolutionScalerConfiguration.isSupported
    let lowLatencyScaleFactors = lowLatencySupported
      ? VTLowLatencySuperResolutionScalerConfiguration.supportedScaleFactors(
        frameWidth: sourcePixelSize.width,
        frameHeight: sourcePixelSize.height
      ).map(Double.init)
      : []

    return AppleUpscalingCapabilitySnapshot(
      sourceKind: sourceKind,
      sourcePixelSize: sourcePixelSize,
      highQualitySupported: highQualitySupported,
      highQualityScaleFactors: highQualityScaleFactors,
      highQualityModelStatus: highQualityModelStatus,
      lowLatencySupported: lowLatencySupported,
      lowLatencyScaleFactors: lowLatencyScaleFactors,
      metalFXSupported: metalFXSupported
    )
  }

  @available(macOS 26.0, *)
  static func modelStatus(
    _ status: VTSuperResolutionScalerConfiguration.ModelStatus
  ) -> UpscalingModelStatus {
    switch status {
    case .downloadRequired: .downloadRequired
    case .downloading: .downloading
    case .ready: .ready
      @unknown default: .downloadRequired
    }
  }

  private static func configurationDimensionCandidates(
    sourceKind: MediaKind,
    sourcePixelSize: UpscalingPixelSize
  ) -> [UpscalingPixelSize] {
    guard sourceKind == .video, sourcePixelSize.width != sourcePixelSize.height else {
      return [sourcePixelSize]
    }
    return [
      sourcePixelSize,
      UpscalingPixelSize(width: sourcePixelSize.height, height: sourcePixelSize.width),
    ]
  }
}
