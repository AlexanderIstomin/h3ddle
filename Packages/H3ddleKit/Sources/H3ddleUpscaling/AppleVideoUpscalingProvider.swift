import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import H3ddleCore

public struct UpscalingVideoProperties: Hashable, Sendable {
  public var pixelSize: UpscalingPixelSize
  public var duration: TimeInterval
  public var hasAudio: Bool

  public init(
    pixelSize: UpscalingPixelSize,
    duration: TimeInterval,
    hasAudio: Bool
  ) {
    self.pixelSize = pixelSize
    self.duration = max(0, duration)
    self.hasAudio = hasAudio
  }
}

public enum UpscalingVideoProbe {
  public static func inspect(_ url: URL) async throws -> UpscalingVideoProperties {
    guard url.isFileURL, FileManager.default.isReadableFile(atPath: url.path) else {
      throw UpscalingError.sourceNotReadable
    }
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw UpscalingError.sourceNotReadable
    }
    let naturalSize = try await track.load(.naturalSize)
    let preferredTransform = try await track.load(.preferredTransform)
    let pixelSize = VideoUpscalingGeometry.displayPixelSize(
      naturalSize: naturalSize,
      preferredTransform: preferredTransform
    )
    let duration = try await asset.load(.duration)
    let seconds = duration.seconds
    guard pixelSize.isValid, seconds.isFinite, seconds > 0 else {
      throw UpscalingError.sourceNotReadable
    }
    let hasAudio = try await !asset.loadTracks(withMediaType: .audio).isEmpty
    return UpscalingVideoProperties(
      pixelSize: pixelSize,
      duration: seconds,
      hasAudio: hasAudio
    )
  }
}

public struct AppleVideoUpscalingProvider: UpscalingProvider {
  private let capabilityProbe: @Sendable (
    MediaKind,
    UpscalingPixelSize
  ) -> AppleUpscalingCapabilitySnapshot

  public init() {
    capabilityProbe = { kind, size in
      AppleUpscalingCapabilityProbe.inspect(sourceKind: kind, sourcePixelSize: size)
    }
  }

  init(
    capabilityProbe: @escaping @Sendable (
      MediaKind,
      UpscalingPixelSize
    ) -> AppleUpscalingCapabilitySnapshot
  ) {
    self.capabilityProbe = capabilityProbe
  }

  public func capabilities(for request: UpscalingRequest) async -> [UpscalingCapability] {
    guard request.sourceKind == .video else { return [] }
    let snapshot = capabilityProbe(request.sourceKind, request.sourcePixelSize)
    var capabilities: [UpscalingCapability] = []
    if snapshot.highQualitySupported, !snapshot.highQualityScaleFactors.isEmpty {
      capabilities.append(
        UpscalingCapability(
          backend: .appleVideoToolbox,
          displayName: "Apple Temporal Super Resolution",
          supportedMediaKinds: [.video],
          supportedScaleFactors: snapshot.highQualityScaleFactors,
          supportsArbitraryOutputSize: false,
          usesTemporalFrames: true,
          availability: highQualityAvailability(snapshot.highQualityModelStatus)
        )
      )
    }
    capabilities.append(
      UpscalingCapability(
        backend: .avFoundation,
        displayName: "AVFoundation High Quality",
        supportedMediaKinds: [.video],
        supportsArbitraryOutputSize: true,
        usesTemporalFrames: false,
        availability: .ready
      )
    )
    return capabilities
  }

  public func events(
    for request: UpscalingRequest
  ) -> AsyncThrowingStream<UpscalingEvent, any Error> {
    AsyncThrowingStream { continuation in
      let capabilityProbe = self.capabilityProbe
      let task = Task.detached(priority: .userInitiated) {
        do {
          try request.validate()
          guard request.sourceKind == .video else {
            throw UpscalingError.unsupportedMediaKind
          }
          let plan: AppleVideoUpscalingPlan
          if request.mode == .fast {
            // Fast mode always uses the model-free AVFoundation path. Avoid
            // initializing the optional temporal model and GPU driver when
            // their capabilities cannot affect backend selection.
            plan = AppleVideoUpscalingPlan(
              backend: .avFoundation,
              videoToolboxScaleFactor: nil
            )
          } else {
            let snapshot = capabilityProbe(request.sourceKind, request.sourcePixelSize)
            plan = try AppleVideoUpscalingPlanner.plan(for: request, snapshot: snapshot)
          }
          continuation.yield(.preparing(backend: plan.backend))
          continuation.yield(.progress(phase: "Reading video", fractionComplete: 0.05))
          let output: UpscalingResult
          switch plan.backend {
          case .appleVideoToolbox:
            output = try await AppleTemporalVideoUpscalingPipeline.process(
              request: request,
              scaleFactor: plan.videoToolboxScaleFactor
            ) { fractionComplete in
              continuation.yield(
                .progress(
                  phase: "Upscaling video",
                  fractionComplete: 0.05 + min(max(fractionComplete, 0), 1) * 0.88
                )
              )
            }
          case .avFoundation:
            continuation.yield(.progress(phase: "Upscaling video", fractionComplete: 0.3))
            output = try await AppleAVFoundationVideoUpscalingPipeline.process(request: request)
          default:
            throw UpscalingError.unavailable("The selected video upscaler is unavailable.")
          }
          try Task.checkCancellation()
          continuation.yield(.progress(phase: "Writing asset", fractionComplete: 1))
          continuation.yield(.completed(output))
          continuation.finish()
        } catch is CancellationError {
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in task.cancel() }
    }
  }

  private func highQualityAvailability(
    _ status: UpscalingModelStatus?
  ) -> UpscalingAvailability {
    switch status {
    case .ready, .notRequired:
      .ready
    case .downloadRequired, .downloading:
      .modelDownloadRequired
    case nil:
      .unavailable(reason: "The source dimensions are not supported.")
    }
  }
}

struct AppleVideoUpscalingPlan: Equatable, Sendable {
  var backend: UpscalingBackendID
  var videoToolboxScaleFactor: Int?
}

enum AppleVideoUpscalingPlanner {
  static func plan(
    for request: UpscalingRequest,
    snapshot: AppleUpscalingCapabilitySnapshot
  ) throws -> AppleVideoUpscalingPlan {
    let requiredFactor = max(
      Double(request.targetPixelSize.width) / Double(request.sourcePixelSize.width),
      Double(request.targetPixelSize.height) / Double(request.sourcePixelSize.height)
    )
    let scaleFactor = snapshot.highQualityScaleFactors
      .filter { $0 >= requiredFactor }
      .min()
      .map(Int.init)
    let highQualityReady = snapshot.highQualitySupported
      && snapshot.highQualityModelStatus == .ready
      && scaleFactor != nil

    switch request.mode {
    case .best where highQualityReady:
      return AppleVideoUpscalingPlan(
        backend: .appleVideoToolbox,
        videoToolboxScaleFactor: scaleFactor
      )
    case .detailed:
      if snapshot.highQualityModelStatus == .downloadRequired
        || snapshot.highQualityModelStatus == .downloading
      {
        throw UpscalingError.modelDownloadRequired
      }
      guard highQualityReady else { throw UpscalingError.unsupportedScaleFactor }
      return AppleVideoUpscalingPlan(
        backend: .appleVideoToolbox,
        videoToolboxScaleFactor: scaleFactor
      )
    case .best, .fast:
      return AppleVideoUpscalingPlan(
        backend: .avFoundation,
        videoToolboxScaleFactor: nil
      )
    }
  }
}

enum VideoUpscalingGeometry {
  static func displayPixelSize(
    naturalSize: CGSize,
    preferredTransform: CGAffineTransform
  ) -> UpscalingPixelSize {
    let rect = CGRect(origin: .zero, size: naturalSize).applying(preferredTransform)
    return UpscalingPixelSize(
      width: Int(abs(rect.width).rounded()),
      height: Int(abs(rect.height).rounded())
    )
  }

  static func renderTransform(
    naturalSize: CGSize,
    preferredTransform: CGAffineTransform,
    targetSize: UpscalingPixelSize
  ) -> CGAffineTransform {
    let displayRect = CGRect(origin: .zero, size: naturalSize).applying(preferredTransform)
    let normalized = preferredTransform.concatenating(
      CGAffineTransform(translationX: -displayRect.minX, y: -displayRect.minY)
    )
    let scale = CGAffineTransform(
      scaleX: CGFloat(targetSize.width) / abs(displayRect.width),
      y: CGFloat(targetSize.height) / abs(displayRect.height)
    )
    return normalized.concatenating(scale)
  }

  static func encodedPixelSize(
    naturalSize: CGSize,
    preferredTransform: CGAffineTransform,
    targetDisplaySize: UpscalingPixelSize
  ) -> UpscalingPixelSize {
    let sourceDisplaySize = displayPixelSize(
      naturalSize: naturalSize,
      preferredTransform: preferredTransform
    )
    guard sourceDisplaySize.isValid else {
      return UpscalingPixelSize(width: 0, height: 0)
    }
    return UpscalingPixelSize(
      width: Int(
        (abs(naturalSize.width) * CGFloat(targetDisplaySize.width)
          / CGFloat(sourceDisplaySize.width)).rounded()
      ),
      height: Int(
        (abs(naturalSize.height) * CGFloat(targetDisplaySize.height)
          / CGFloat(sourceDisplaySize.height)).rounded()
      )
    )
  }

  static func outputPreferredTransform(
    encodedPixelSize: UpscalingPixelSize,
    sourcePreferredTransform: CGAffineTransform
  ) -> CGAffineTransform {
    let orientation = CGAffineTransform(
      a: sourcePreferredTransform.a,
      b: sourcePreferredTransform.b,
      c: sourcePreferredTransform.c,
      d: sourcePreferredTransform.d,
      tx: 0,
      ty: 0
    )
    let displayRect = CGRect(
      x: 0,
      y: 0,
      width: encodedPixelSize.width,
      height: encodedPixelSize.height
    ).applying(orientation)
    return orientation.concatenating(
      CGAffineTransform(translationX: -displayRect.minX, y: -displayRect.minY)
    )
  }
}

enum AppleAVFoundationVideoUpscalingPipeline {
  static func process(request: UpscalingRequest) async throws -> UpscalingResult {
    let fileManager = FileManager.default
    guard !fileManager.fileExists(atPath: request.destinationURL.path) else {
      throw UpscalingError.destinationAlreadyExists
    }
    try fileManager.createDirectory(
      at: request.destinationURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let asset = AVURLAsset(url: request.sourceURL)
    guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
      throw UpscalingError.sourceNotReadable
    }
    let duration = try await asset.load(.duration)
    guard duration.isValid, duration.seconds.isFinite, duration.seconds > 0 else {
      throw UpscalingError.sourceNotReadable
    }
    let naturalSize = try await videoTrack.load(.naturalSize)
    let preferredTransform = try await videoTrack.load(.preferredTransform)
    let actualSize = VideoUpscalingGeometry.displayPixelSize(
      naturalSize: naturalSize,
      preferredTransform: preferredTransform
    )
    guard actualSize == request.sourcePixelSize else {
      throw UpscalingError.sourceDimensionsMismatch(
        expected: request.sourcePixelSize,
        actual: actualSize
      )
    }

    guard let export = AVAssetExportSession(
      asset: asset,
      presetName: AVAssetExportPresetHighestQuality
    ) else {
      throw UpscalingError.unavailable("AVFoundation cannot export this video.")
    }
    export.videoComposition = try await videoComposition(
      track: videoTrack,
      duration: duration,
      naturalSize: naturalSize,
      preferredTransform: preferredTransform,
      targetSize: request.targetPixelSize
    )
    export.shouldOptimizeForNetworkUse = true
    export.allowsParallelizedExport = true
    let fileType: AVFileType = request.destinationURL.pathExtension.lowercased() == "mov"
      ? .mov
      : .mp4
    var completed = false
    defer {
      if !completed { try? fileManager.removeItem(at: request.destinationURL) }
    }
    try await export.export(to: request.destinationURL, as: fileType)
    completed = true

    return UpscalingResult(
      requestID: request.id,
      outputURL: request.destinationURL,
      mediaKind: .video,
      pixelSize: request.targetPixelSize,
      duration: duration.seconds
    )
  }

  private static func videoComposition(
    track: AVAssetTrack,
    duration: CMTime,
    naturalSize: CGSize,
    preferredTransform: CGAffineTransform,
    targetSize: UpscalingPixelSize
  ) async throws -> AVMutableVideoComposition {
    let nominalFrameRate = try await track.load(.nominalFrameRate)
    let framesPerSecond = max(1, Int32(nominalFrameRate.rounded()))
    let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: track)
    layerInstruction.setTransform(
      VideoUpscalingGeometry.renderTransform(
        naturalSize: naturalSize,
        preferredTransform: preferredTransform,
        targetSize: targetSize
      ),
      at: .zero
    )
    let instruction = AVMutableVideoCompositionInstruction()
    instruction.timeRange = CMTimeRange(start: .zero, duration: duration)
    instruction.layerInstructions = [layerInstruction]

    let composition = AVMutableVideoComposition()
    composition.renderSize = CGSize(width: targetSize.width, height: targetSize.height)
    composition.frameDuration = CMTime(value: 1, timescale: framesPerSecond)
    composition.instructions = [instruction]
    return composition
  }
}
