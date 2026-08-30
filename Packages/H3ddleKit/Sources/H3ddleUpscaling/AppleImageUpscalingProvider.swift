import Accelerate
import CoreImage
import CoreMedia
import CoreVideo
import Foundation
import H3ddleCore
import ImageIO
import UniformTypeIdentifiers
import VideoToolbox

public struct AppleImageUpscalingProvider: UpscalingProvider {
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
    guard request.sourceKind == .image else { return [] }
    let snapshot = capabilityProbe(request.sourceKind, request.sourcePixelSize)
    var capabilities: [UpscalingCapability] = []

    if snapshot.highQualitySupported, !snapshot.highQualityScaleFactors.isEmpty {
      capabilities.append(
        UpscalingCapability(
          backend: .appleVideoToolbox,
          displayName: "Apple Super Resolution",
          supportedMediaKinds: [.image],
          supportedScaleFactors: snapshot.highQualityScaleFactors,
          supportsArbitraryOutputSize: false,
          usesTemporalFrames: false,
          availability: highQualityAvailability(snapshot.highQualityModelStatus)
        )
      )
    }

    capabilities.append(
      UpscalingCapability(
        backend: .accelerateVImage,
        displayName: "Accelerate High Quality",
        supportedMediaKinds: [.image],
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
          guard request.sourceKind == .image else {
            throw UpscalingError.unsupportedMediaKind
          }
          let plan: AppleImageUpscalingPlan
          if request.mode == .fast {
            // Fast mode always uses the model-free Accelerate path. Avoid
            // waking the optional VideoToolbox model (and its GPU driver) when
            // its result cannot affect backend selection.
            plan = AppleImageUpscalingPlan(
              backend: .accelerateVImage,
              videoToolboxScaleFactor: nil
            )
          } else {
            let snapshot = capabilityProbe(request.sourceKind, request.sourcePixelSize)
            plan = try AppleImageUpscalingPlanner.plan(for: request, snapshot: snapshot)
          }

          continuation.yield(.preparing(backend: plan.backend))
          continuation.yield(.progress(phase: "Reading source", fractionComplete: 0.1))
          try Task.checkCancellation()
          continuation.yield(.progress(phase: "Upscaling image", fractionComplete: 0.35))

          let output = try await AppleImageUpscalingPipeline.process(
            request: request,
            plan: plan
          )
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
    case .ready:
      .ready
    case .downloadRequired, .downloading:
      .modelDownloadRequired
    case .notRequired:
      .ready
    case nil:
      .unavailable(reason: "The source dimensions are not supported.")
    }
  }
}

struct AppleImageUpscalingPlan: Equatable, Sendable {
  var backend: UpscalingBackendID
  var videoToolboxScaleFactor: Int?
}

enum AppleImageUpscalingPlanner {
  static func plan(
    for request: UpscalingRequest,
    snapshot: AppleUpscalingCapabilitySnapshot
  ) throws -> AppleImageUpscalingPlan {
    let requiredFactor = max(
      Double(request.targetPixelSize.width) / Double(request.sourcePixelSize.width),
      Double(request.targetPixelSize.height) / Double(request.sourcePixelSize.height)
    )
    let scaleFactor = snapshot.highQualityScaleFactors
      .filter { $0 >= requiredFactor }
      .sorted()
      .first
      .map { Int($0) }
    let highQualityReady = snapshot.highQualitySupported
      && snapshot.highQualityModelStatus == .ready
      && scaleFactor != nil

    switch request.mode {
    case .best where highQualityReady:
      return AppleImageUpscalingPlan(
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
      return AppleImageUpscalingPlan(
        backend: .appleVideoToolbox,
        videoToolboxScaleFactor: scaleFactor
      )
    case .best, .fast:
      return AppleImageUpscalingPlan(
        backend: .accelerateVImage,
        videoToolboxScaleFactor: nil
      )
    }
  }
}

private enum AppleImageUpscalingPipeline {
  static func process(
    request: UpscalingRequest,
    plan: AppleImageUpscalingPlan
  ) async throws -> UpscalingResult {
    guard !FileManager.default.fileExists(atPath: request.destinationURL.path) else {
      throw UpscalingError.destinationAlreadyExists
    }
    let source = try readSource(at: request.sourceURL)
    let actualSize = UpscalingPixelSize(width: source.width, height: source.height)
    guard actualSize == request.sourcePixelSize else {
      throw UpscalingError.sourceDimensionsMismatch(
        expected: request.sourcePixelSize,
        actual: actualSize
      )
    }

    let processed: CGImage
    switch plan.backend {
    case .appleVideoToolbox:
      guard let factor = plan.videoToolboxScaleFactor else {
        throw UpscalingError.unsupportedScaleFactor
      }
      let upscaled = try await videoToolboxUpscale(
        CIImage(cgImage: source),
        sourceSize: actualSize,
        scaleFactor: factor
      )
      processed = try render(
        upscaled,
        pixelSize: UpscalingPixelSize(
          width: actualSize.width * factor,
          height: actualSize.height * factor
        )
      )
    case .accelerateVImage:
      processed = source
    default:
      throw UpscalingError.unavailable("The selected image upscaling backend is unavailable.")
    }

    try Task.checkCancellation()
    let outputImage = try resize(processed, to: request.targetPixelSize)
    try write(outputImage, to: request.destinationURL)

    return UpscalingResult(
      requestID: request.id,
      outputURL: request.destinationURL,
      mediaKind: .image,
      pixelSize: request.targetPixelSize,
      duration: request.sourceDuration
    )
  }

  private static func readSource(at url: URL) throws -> CGImage {
    let options = [kCGImageSourceShouldCacheImmediately: true] as CFDictionary
    guard FileManager.default.isReadableFile(atPath: url.path),
      let source = CGImageSourceCreateWithURL(url as CFURL, options),
      let image = CGImageSourceCreateImageAtIndex(source, 0, options)
    else {
      throw UpscalingError.sourceNotReadable
    }
    return image
  }

  private static func resize(
    _ image: CGImage,
    to target: UpscalingPixelSize
  ) throws -> CGImage {
    guard image.width != target.width || image.height != target.height else { return image }
    let colorSpace = image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!
    var format = vImage_CGImageFormat(
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      colorSpace: Unmanaged.passUnretained(colorSpace),
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
        .union(.byteOrder32Big),
      version: 0,
      decode: nil,
      renderingIntent: .defaultIntent
    )
    var sourceBuffer = vImage_Buffer()
    var error = vImageBuffer_InitWithCGImage(
      &sourceBuffer,
      &format,
      nil,
      image,
      vImage_Flags(kvImageNoFlags)
    )
    guard error == kvImageNoError else {
      throw UpscalingError.failed("Accelerate could not decode the source image (\(error)).")
    }
    defer { free(sourceBuffer.data) }

    var destinationBuffer = vImage_Buffer()
    error = vImageBuffer_Init(
      &destinationBuffer,
      vImagePixelCount(target.height),
      vImagePixelCount(target.width),
      format.bitsPerPixel,
      vImage_Flags(kvImageNoFlags)
    )
    guard error == kvImageNoError else {
      throw UpscalingError.failed("Accelerate could not allocate the upscale output (\(error)).")
    }
    defer { free(destinationBuffer.data) }

    error = vImageScale_ARGB8888(
      &sourceBuffer,
      &destinationBuffer,
      nil,
      vImage_Flags(kvImageHighQualityResampling)
    )
    guard error == kvImageNoError else {
      throw UpscalingError.failed("Accelerate could not resize the source image (\(error)).")
    }
    var createError = kvImageNoError
    guard let output = vImageCreateCGImageFromBuffer(
      &destinationBuffer,
      &format,
      nil,
      nil,
      vImage_Flags(kvImageNoFlags),
      &createError
    )?.takeRetainedValue(), createError == kvImageNoError else {
      throw UpscalingError.failed("Accelerate could not create the upscaled image (\(createError)).")
    }
    return output
  }

  private static func render(
    _ image: CIImage,
    pixelSize: UpscalingPixelSize
  ) throws -> CGImage {
    let context = CIContext(options: [.cacheIntermediates: false])
    let bounds = CGRect(x: 0, y: 0, width: pixelSize.width, height: pixelSize.height)
    guard let output = context.createCGImage(image, from: bounds) else {
      throw UpscalingError.failed(
        "Core Image could not render the upscaled image (extent: \(image.extent), bounds: \(bounds))."
      )
    }
    return output
  }

  private static func write(_ image: CGImage, to destinationURL: URL) throws {
    let fileManager = FileManager.default
    let directory = destinationURL.deletingLastPathComponent()
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    let fileType = UTType(filenameExtension: destinationURL.pathExtension)
      .flatMap { $0.conforms(to: .image) ? $0 : nil } ?? .png
    let temporaryURL = directory.appendingPathComponent(
      ".\(UUID().uuidString).\(destinationURL.pathExtension.isEmpty ? "png" : destinationURL.pathExtension)"
    )
    defer { try? fileManager.removeItem(at: temporaryURL) }

    guard let destination = CGImageDestinationCreateWithURL(
      temporaryURL as CFURL,
      fileType.identifier as CFString,
      1,
      nil
    ) else {
      throw UpscalingError.failed("The upscale destination could not be created.")
    }
    let properties: CFDictionary = [
      kCGImageDestinationLossyCompressionQuality: 0.95
    ] as CFDictionary
    CGImageDestinationAddImage(destination, image, properties)
    guard CGImageDestinationFinalize(destination) else {
      throw UpscalingError.failed("The upscaled image could not be encoded.")
    }
    try fileManager.moveItem(at: temporaryURL, to: destinationURL)
  }

  @available(macOS 26.0, *)
  private static func videoToolboxUpscaleAvailable(
    _ image: CIImage,
    sourceSize: UpscalingPixelSize,
    scaleFactor: Int
  ) async throws -> CIImage {
    guard let configuration = VTSuperResolutionScalerConfiguration(
      frameWidth: sourceSize.width,
      frameHeight: sourceSize.height,
      scaleFactor: scaleFactor,
      inputType: .image,
      usePrecomputedFlow: false,
      qualityPrioritization: .normal,
      revision: VTSuperResolutionScalerConfiguration.defaultRevision
    ) else {
      throw UpscalingError.unsupportedScaleFactor
    }
    guard configuration.configurationModelStatus == .ready else {
      throw UpscalingError.modelDownloadRequired
    }

    let outputSize = UpscalingPixelSize(
      width: sourceSize.width * scaleFactor,
      height: sourceSize.height * scaleFactor
    )
    let sourceBuffer = try makeHalfFloatPixelBuffer(size: sourceSize)
    let destinationBuffer = try makeHalfFloatPixelBuffer(size: outputSize)
    let context = CIContext(options: [.cacheIntermediates: false])
    context.render(
      image,
      to: sourceBuffer,
      bounds: CGRect(x: 0, y: 0, width: sourceSize.width, height: sourceSize.height),
      colorSpace: image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)
    )

    guard let sourceFrame = VTFrameProcessorFrame(
      buffer: sourceBuffer,
      presentationTimeStamp: .zero
    ), let destinationFrame = VTFrameProcessorFrame(
      buffer: destinationBuffer,
      presentationTimeStamp: .zero
    ), let parameters = VTSuperResolutionScalerParameters(
      sourceFrame: sourceFrame,
      previousFrame: nil,
      previousOutputFrame: nil,
      opticalFlow: nil,
      submissionMode: .random,
      destinationFrame: destinationFrame
    ) else {
      throw UpscalingError.failed("VideoToolbox rejected the image frame buffers.")
    }

    let processor = VTFrameProcessor()
    try processor.startSession(configuration: configuration)
    defer { processor.endSession() }
    try await withCheckedThrowingContinuation {
      (continuation: CheckedContinuation<Void, any Error>) in
      processor.process(parameters: parameters) { _, error in
        if let error {
          continuation.resume(throwing: error)
        } else {
          continuation.resume()
        }
      }
    }
    return CIImage(cvPixelBuffer: destinationBuffer)
  }

  private static func videoToolboxUpscale(
    _ image: CIImage,
    sourceSize: UpscalingPixelSize,
    scaleFactor: Int
  ) async throws -> CIImage {
    guard #available(macOS 26.0, *) else {
      throw UpscalingError.unavailable("Apple Super Resolution requires macOS 26 or later.")
    }
    return try await videoToolboxUpscaleAvailable(
      image,
      sourceSize: sourceSize,
      scaleFactor: scaleFactor
    )
  }

  private static func makeHalfFloatPixelBuffer(
    size: UpscalingPixelSize
  ) throws -> CVPixelBuffer {
    let attributes = [
      kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
      kCVPixelBufferMetalCompatibilityKey: true as CFBoolean,
    ] as CFDictionary
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      size.width,
      size.height,
      kCVPixelFormatType_64RGBAHalf,
      attributes,
      &buffer
    )
    guard status == kCVReturnSuccess, let buffer else {
      throw UpscalingError.failed("A VideoToolbox-compatible pixel buffer could not be allocated.")
    }
    return buffer
  }
}
