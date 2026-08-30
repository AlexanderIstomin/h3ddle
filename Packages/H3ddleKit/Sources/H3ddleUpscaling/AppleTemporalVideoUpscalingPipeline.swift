import AVFoundation
import CoreImage
import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

enum AppleTemporalVideoUpscalingPipeline {
  static func process(
    request: UpscalingRequest,
    scaleFactor: Int?,
    progress: @escaping @Sendable (Double) -> Void
  ) async throws -> UpscalingResult {
    guard #available(macOS 26.0, *) else {
      throw UpscalingError.unavailable(
        "Apple Temporal Super Resolution requires macOS 26 or later."
      )
    }
    guard let scaleFactor else { throw UpscalingError.unsupportedScaleFactor }
    return try await processAvailable(
      request: request,
      scaleFactor: scaleFactor,
      progress: progress
    )
  }

  @available(macOS 26.0, *)
  private static func processAvailable(
    request: UpscalingRequest,
    scaleFactor: Int,
    progress: @escaping @Sendable (Double) -> Void
  ) async throws -> UpscalingResult {
    let fileManager = FileManager.default
    guard !fileManager.fileExists(atPath: request.destinationURL.path) else {
      throw UpscalingError.destinationAlreadyExists
    }
    let directory = request.destinationURL.deletingLastPathComponent()
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    let temporaryVideoURL = directory.appendingPathComponent(
      ".\(UUID().uuidString.lowercased())-temporal.mov"
    )
    var completed = false
    defer {
      try? fileManager.removeItem(at: temporaryVideoURL)
      if !completed { try? fileManager.removeItem(at: request.destinationURL) }
    }

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
    let actualDisplaySize = VideoUpscalingGeometry.displayPixelSize(
      naturalSize: naturalSize,
      preferredTransform: preferredTransform
    )
    guard actualDisplaySize == request.sourcePixelSize else {
      throw UpscalingError.sourceDimensionsMismatch(
        expected: request.sourcePixelSize,
        actual: actualDisplaySize
      )
    }

    let encodedSourceSize = UpscalingPixelSize(
      width: Int(abs(naturalSize.width).rounded()),
      height: Int(abs(naturalSize.height).rounded())
    )
    let encodedTargetSize = VideoUpscalingGeometry.encodedPixelSize(
      naturalSize: naturalSize,
      preferredTransform: preferredTransform,
      targetDisplaySize: request.targetPixelSize
    )
    guard encodedSourceSize.isValid, encodedTargetSize.isValid else {
      throw UpscalingError.invalidDimensions
    }
    guard let configuration = VTSuperResolutionScalerConfiguration(
      frameWidth: encodedSourceSize.width,
      frameHeight: encodedSourceSize.height,
      scaleFactor: scaleFactor,
      inputType: .video,
      usePrecomputedFlow: false,
      qualityPrioritization: .normal,
      revision: VTSuperResolutionScalerConfiguration.defaultRevision
    ) else {
      throw UpscalingError.unavailable(
        "Apple Temporal Super Resolution does not support this video's encoded dimensions."
      )
    }
    guard configuration.configurationModelStatus == .ready else {
      throw UpscalingError.modelDownloadRequired
    }
    guard configuration.supportedPixelFormats.contains(kCVPixelFormatType_64RGBAHalf) else {
      throw UpscalingError.unavailable(
        "Apple Temporal Super Resolution requires an unsupported pixel format."
      )
    }

    let reader = try AVAssetReader(asset: asset)
    let readerOutput = AVAssetReaderTrackOutput(
      track: videoTrack,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      ]
    )
    readerOutput.alwaysCopiesSampleData = false
    guard reader.canAdd(readerOutput) else {
      throw UpscalingError.failed("AVFoundation could not prepare the video decoder.")
    }
    reader.add(readerOutput)

    let nominalFrameRate = max(1, try await videoTrack.load(.nominalFrameRate))
    let writer = try AVAssetWriter(outputURL: temporaryVideoURL, fileType: .mov)
    let writerInput = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: videoOutputSettings(
        size: encodedTargetSize,
        nominalFrameRate: nominalFrameRate
      )
    )
    writerInput.expectsMediaDataInRealTime = false
    writerInput.transform = VideoUpscalingGeometry.outputPreferredTransform(
      encodedPixelSize: encodedTargetSize,
      sourcePreferredTransform: preferredTransform
    )
    guard writer.canAdd(writerInput) else {
      throw UpscalingError.failed("AVFoundation rejected the temporal video encode settings.")
    }
    writer.add(writerInput)
    let writerAdaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: writerInput,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: encodedTargetSize.width,
        kCVPixelBufferHeightKey as String: encodedTargetSize.height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      ]
    )

    guard writer.startWriting() else {
      throw UpscalingError.failed(
        writer.error?.localizedDescription ?? "The temporal video writer could not start."
      )
    }
    writer.startSession(atSourceTime: .zero)
    guard reader.startReading() else {
      writer.cancelWriting()
      throw UpscalingError.failed(
        reader.error?.localizedDescription ?? "The source video could not be decoded."
      )
    }
    var writerFinished = false
    defer {
      if reader.status == .reading { reader.cancelReading() }
      if !writerFinished { writer.cancelWriting() }
    }

    let processor = VTFrameProcessor()
    try processor.startSession(configuration: configuration)
    defer { processor.endSession() }
    let context = CIContext(options: [.cacheIntermediates: false])
    let modelOutputSize = UpscalingPixelSize(
      width: encodedSourceSize.width * scaleFactor,
      height: encodedSourceSize.height * scaleFactor
    )
    var previousSourceFrame: VTFrameProcessorFrame?
    var previousOutputFrame: VTFrameProcessorFrame?
    var firstPresentationTime: CMTime?
    var frameCount = 0

    while reader.status == .reading {
      try Task.checkCancellation()
      guard let sampleBuffer = readerOutput.copyNextSampleBuffer() else { break }
      guard let decodedBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        throw UpscalingError.failed("The video decoder returned a frame without pixels.")
      }
      let sourcePresentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
      if firstPresentationTime == nil { firstPresentationTime = sourcePresentationTime }
      let presentationTime = CMTimeSubtract(
        sourcePresentationTime,
        firstPresentationTime ?? sourcePresentationTime
      )

      let sourceBuffer = try makePixelBuffer(size: encodedSourceSize)
      let modelOutputBuffer = try makePixelBuffer(size: modelOutputSize)
      let sourceImage = CIImage(cvPixelBuffer: decodedBuffer)
      context.render(
        sourceImage,
        to: sourceBuffer,
        bounds: CGRect(
          x: 0,
          y: 0,
          width: encodedSourceSize.width,
          height: encodedSourceSize.height
        ),
        colorSpace: sourceImage.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)
      )

      guard let sourceFrame = VTFrameProcessorFrame(
        buffer: sourceBuffer,
        presentationTimeStamp: presentationTime
      ), let outputFrame = VTFrameProcessorFrame(
        buffer: modelOutputBuffer,
        presentationTimeStamp: presentationTime
      ), let parameters = VTSuperResolutionScalerParameters(
        sourceFrame: sourceFrame,
        previousFrame: previousSourceFrame,
        previousOutputFrame: previousOutputFrame,
        opticalFlow: nil,
        submissionMode: previousSourceFrame == nil ? .random : .sequential,
        destinationFrame: outputFrame
      ) else {
        throw UpscalingError.failed("VideoToolbox rejected a temporal video frame.")
      }
      try await process(parameters: parameters, with: processor)
      try Task.checkCancellation()
      try await waitUntilReady(writerInput)
      guard let writerPool = writerAdaptor.pixelBufferPool else {
        throw UpscalingError.failed("The video writer did not create a pixel-buffer pool.")
      }
      var writerBuffer: CVPixelBuffer?
      let allocationStatus = CVPixelBufferPoolCreatePixelBuffer(
        kCFAllocatorDefault,
        writerPool,
        &writerBuffer
      )
      guard allocationStatus == kCVReturnSuccess, let writerBuffer else {
        throw UpscalingError.failed("The video writer could not allocate an output frame.")
      }
      let modelImage = CIImage(cvPixelBuffer: modelOutputBuffer)
      let fittedImage = modelImage.transformed(
        by: CGAffineTransform(
          scaleX: CGFloat(encodedTargetSize.width) / CGFloat(modelOutputSize.width),
          y: CGFloat(encodedTargetSize.height) / CGFloat(modelOutputSize.height)
        )
      )
      context.render(
        fittedImage,
        to: writerBuffer,
        bounds: CGRect(
          x: 0,
          y: 0,
          width: encodedTargetSize.width,
          height: encodedTargetSize.height
        ),
        colorSpace: modelImage.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)
      )
      guard writerAdaptor.append(writerBuffer, withPresentationTime: presentationTime) else {
        throw UpscalingError.failed(
          writer.error?.localizedDescription ?? "The video writer rejected an upscaled frame."
        )
      }
      previousSourceFrame = sourceFrame
      previousOutputFrame = outputFrame
      frameCount += 1
      let elapsed = presentationTime.seconds
      if elapsed.isFinite, duration.seconds > 0 {
        progress(min(max(elapsed / duration.seconds, 0), 1))
      }
    }
    try checkReader(reader)
    guard frameCount > 0 else { throw UpscalingError.sourceNotReadable }
    writerInput.markAsFinished()
    await writer.finishWriting()
    guard writer.status == .completed else {
      throw UpscalingError.failed(
        writer.error?.localizedDescription ?? "The temporal video could not be finalized."
      )
    }
    writerFinished = true
    progress(0.97)

    try await writeFinalAsset(
      processedVideoURL: temporaryVideoURL,
      sourceAsset: asset,
      request: request
    )
    completed = true
    return UpscalingResult(
      requestID: request.id,
      outputURL: request.destinationURL,
      mediaKind: .video,
      pixelSize: request.targetPixelSize,
      duration: duration.seconds
    )
  }

  @available(macOS 26.0, *)
  private static func process(
    parameters: VTSuperResolutionScalerParameters,
    with processor: VTFrameProcessor
  ) async throws {
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
  }

  private static func videoOutputSettings(
    size: UpscalingPixelSize,
    nominalFrameRate: Float
  ) -> [String: Any] {
    let pixelRate = Double(size.width) * Double(size.height) * Double(nominalFrameRate)
    let bitRate = Int(min(max(pixelRate * 0.12, 2_000_000), 120_000_000))
    let codec: AVVideoCodecType = size.width > 4_096 || size.height > 2_304 ? .hevc : .h264
    return [
      AVVideoCodecKey: codec,
      AVVideoWidthKey: size.width,
      AVVideoHeightKey: size.height,
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: bitRate,
        AVVideoExpectedSourceFrameRateKey: nominalFrameRate,
      ],
    ]
  }

  private static func makePixelBuffer(size: UpscalingPixelSize) throws -> CVPixelBuffer {
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
      throw UpscalingError.failed("A temporal VideoToolbox buffer could not be allocated.")
    }
    return buffer
  }

  private static func waitUntilReady(_ input: AVAssetWriterInput) async throws {
    let deadline = ContinuousClock.now + .seconds(30)
    while !input.isReadyForMoreMediaData {
      try Task.checkCancellation()
      if ContinuousClock.now > deadline {
        throw UpscalingError.failed("The video encoder stopped accepting frames.")
      }
      try await Task.sleep(for: .milliseconds(4))
    }
  }

  private static func checkReader(_ reader: AVAssetReader) throws {
    switch reader.status {
    case .completed:
      return
    case .cancelled:
      throw CancellationError()
    case .failed:
      throw UpscalingError.failed(
        reader.error?.localizedDescription ?? "The source video could not be decoded."
      )
    case .unknown, .reading:
      throw UpscalingError.failed("The source video decoder stopped unexpectedly.")
    @unknown default:
      throw UpscalingError.failed("The source video decoder returned an unknown state.")
    }
  }

  static func writeFinalAsset(
    processedVideoURL: URL,
    sourceAsset: AVURLAsset,
    request: UpscalingRequest
  ) async throws {
    let processedAsset = AVURLAsset(url: processedVideoURL)
    guard let processedTrack = try await processedAsset.loadTracks(withMediaType: .video).first
    else {
      throw UpscalingError.failed("The processed video track could not be reopened.")
    }
    let processedDuration = try await processedAsset.load(.duration)
    let composition = AVMutableComposition()
    guard let videoTrack = composition.addMutableTrack(
      withMediaType: .video,
      preferredTrackID: kCMPersistentTrackID_Invalid
    ) else {
      throw UpscalingError.failed("The final video track could not be created.")
    }
    try videoTrack.insertTimeRange(
      CMTimeRange(start: .zero, duration: processedDuration),
      of: processedTrack,
      at: .zero
    )
    videoTrack.preferredTransform = try await processedTrack.load(.preferredTransform)

    if request.preservesAudio,
      let sourceAudioTrack = try await sourceAsset.loadTracks(withMediaType: .audio).first,
      let audioTrack = composition.addMutableTrack(
        withMediaType: .audio,
        preferredTrackID: kCMPersistentTrackID_Invalid
      )
    {
      let audioRange = try await sourceAudioTrack.load(.timeRange)
      let audioDuration = CMTimeMinimum(audioRange.duration, processedDuration)
      if audioDuration > .zero {
        try audioTrack.insertTimeRange(
          CMTimeRange(start: audioRange.start, duration: audioDuration),
          of: sourceAudioTrack,
          at: .zero
        )
      }
    }

    guard let export = AVAssetExportSession(
      asset: composition,
      presetName: AVAssetExportPresetPassthrough
    ) else {
      throw UpscalingError.failed("AVFoundation could not prepare the final upscaled video.")
    }
    export.shouldOptimizeForNetworkUse = true
    let fileType: AVFileType = request.destinationURL.pathExtension.lowercased() == "mov"
      ? .mov
      : .mp4
    guard export.supportedFileTypes.contains(fileType) else {
      throw UpscalingError.unavailable(
        "The upscaled video and its audio cannot be written as \(fileType.rawValue)."
      )
    }
    try await export.export(to: request.destinationURL, as: fileType)
  }
}
