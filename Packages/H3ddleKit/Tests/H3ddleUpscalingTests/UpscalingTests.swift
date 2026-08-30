import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import H3ddleCore
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import H3ddleUpscaling

@Suite("Upscaling contract")
struct UpscalingTests {
  @Test("A valid visual request creates a deterministic fake result")
  func fakeResult() async throws {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-upscale-test-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: directory) }
    let source = directory.appendingPathComponent("source.png")
    let destination = directory.appendingPathComponent("upscaled.png")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try Data("fixture".utf8).write(to: source)

    let request = UpscalingRequest(
      sourceURL: source,
      sourceKind: .image,
      sourcePixelSize: UpscalingPixelSize(width: 256, height: 256),
      sourceDuration: 3,
      targetPixelSize: UpscalingPixelSize(width: 1_024, height: 1_024),
      destinationURL: destination,
      mode: .best
    )
    let provider = FakeUpscalingProvider(stepDelay: .zero)
    var progressCount = 0
    var result: UpscalingResult?

    for try await event in provider.events(for: request) {
      switch event {
      case .preparing(let backend):
        #expect(backend == .fake)
      case .progress(_, let fractionComplete):
        progressCount += 1
        #expect((0...1).contains(fractionComplete))
      case .completed(let completed):
        result = completed
      }
    }

    #expect(progressCount == 3)
    #expect(result?.requestID == request.id)
    #expect(result?.pixelSize == request.targetPixelSize)
    #expect(try Data(contentsOf: destination) == Data("fixture".utf8))
  }

  @Test("Audio and non-upscale requests are rejected")
  func validation() {
    let base = UpscalingRequest(
      sourceURL: URL(fileURLWithPath: "/tmp/source.mov"),
      sourceKind: .video,
      sourcePixelSize: UpscalingPixelSize(width: 640, height: 360),
      sourceDuration: 4,
      targetPixelSize: UpscalingPixelSize(width: 1_280, height: 720),
      destinationURL: URL(fileURLWithPath: "/tmp/output.mov")
    )
    #expect(throws: Never.self) { try base.validate() }

    var audio = base
    audio.sourceKind = .audio
    #expect(throws: UpscalingError.unsupportedMediaKind) { try audio.validate() }

    var smaller = base
    smaller.targetPixelSize = UpscalingPixelSize(width: 320, height: 180)
    #expect(throws: UpscalingError.targetIsNotLarger) { try smaller.validate() }

    var overwriting = base
    overwriting.destinationURL = overwriting.sourceURL
    #expect(throws: UpscalingError.sourceAndDestinationMatch) { try overwriting.validate() }
  }

  @Test("Fake capability discovery excludes audio")
  func fakeCapabilities() async {
    let provider = FakeUpscalingProvider(stepDelay: .zero)
    var request = request(kind: .video)
    let video = await provider.capabilities(for: request)
    #expect(video.count == 1)
    #expect(video[0].usesTemporalFrames)

    request.sourceKind = .audio
    let audio = await provider.capabilities(for: request)
    #expect(audio.isEmpty)
  }

  @Test("Apple probe is read-only and returns no media capability for audio")
  func appleProbeRejectsAudio() {
    let snapshot = AppleUpscalingCapabilityProbe.inspect(
      sourceKind: .audio,
      sourcePixelSize: UpscalingPixelSize(width: 640, height: 360)
    )
    #expect(!snapshot.highQualitySupported)
    #expect(snapshot.highQualityScaleFactors.isEmpty)
    #expect(!snapshot.lowLatencySupported)
    #expect(snapshot.lowLatencyScaleFactors.isEmpty)
    #expect(!snapshot.metalFXSupported)
  }

  @Test("Fast image mode writes an exact-size local image")
  func appleImageProviderWritesOutput() async throws {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-image-upscale-test-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let source = directory.appendingPathComponent("source.png")
    let destination = directory.appendingPathComponent("upscaled.png")
    try writeFixtureImage(to: source, width: 64, height: 48)

    let request = UpscalingRequest(
      sourceURL: source,
      sourceKind: .image,
      sourcePixelSize: UpscalingPixelSize(width: 64, height: 48),
      sourceDuration: 3,
      targetPixelSize: UpscalingPixelSize(width: 128, height: 96),
      destinationURL: destination,
      mode: .fast
    )
    let provider = AppleImageUpscalingProvider(
      capabilityProbe: { _, _ in
        Issue.record("Fast image upscaling should not probe the optional model backend")
        return self.snapshot(
          kind: .image,
          size: request.sourcePixelSize,
          modelStatus: .downloadRequired
        )
      }
    )
    var preparedBackend: UpscalingBackendID?
    var result: UpscalingResult?

    for try await event in provider.events(for: request) {
      switch event {
      case .preparing(let backend): preparedBackend = backend
      case .progress: break
      case .completed(let completed): result = completed
      }
    }

    #expect(preparedBackend == .accelerateVImage)
    #expect(result?.pixelSize == request.targetPixelSize)
    #expect(imageSize(at: destination) == request.targetPixelSize)
  }

  @Test("Best image mode selects Apple Super Resolution only when ready")
  func imageBackendSelection() throws {
    var request = request(kind: .image)
    request.mode = .best
    let ready = snapshot(
      kind: .image,
      size: request.sourcePixelSize,
      modelStatus: .ready
    )
    let readyPlan = try AppleImageUpscalingPlanner.plan(for: request, snapshot: ready)
    #expect(readyPlan.backend == .appleVideoToolbox)
    #expect(readyPlan.videoToolboxScaleFactor == 4)

    let missing = snapshot(
      kind: .image,
      size: request.sourcePixelSize,
      modelStatus: .downloadRequired
    )
    let fallbackPlan = try AppleImageUpscalingPlanner.plan(for: request, snapshot: missing)
    #expect(fallbackPlan.backend == .accelerateVImage)
  }

  @Test("Detailed image mode reports a required model download")
  func detailedModeRequiresModel() {
    var request = request(kind: .image)
    request.mode = .detailed
    let missing = snapshot(
      kind: .image,
      size: request.sourcePixelSize,
      modelStatus: .downloadRequired
    )
    #expect(throws: UpscalingError.modelDownloadRequired) {
      try AppleImageUpscalingPlanner.plan(for: request, snapshot: missing)
    }
  }

  @Test("Model setup chooses the smallest supported factor that covers the target")
  func modelDownloadScaleSelection() {
    #expect(
      AppleSuperResolutionScalePlanner.scaleFactor(
        minimumScaleFactor: 2,
        supportedScaleFactors: [4, 2]
      ) == 2
    )
    #expect(
      AppleSuperResolutionScalePlanner.scaleFactor(
        minimumScaleFactor: 2,
        supportedScaleFactors: [4]
      ) == 4
    )
    #expect(
      AppleSuperResolutionScalePlanner.scaleFactor(
        minimumScaleFactor: 5,
        supportedScaleFactors: [2, 4]
      ) == nil
    )
  }

  @Test("Model download progress is clamped at the public boundary")
  func modelDownloadProgressClamps() {
    #expect(
      UpscalingModelDownloadSnapshot(status: .downloading, fractionComplete: -0.5)
        .fractionComplete == 0
    )
    #expect(
      UpscalingModelDownloadSnapshot(status: .downloading, fractionComplete: 1.5)
        .fractionComplete == 1
    )
  }

  @Test("Video geometry respects portrait orientation and target size")
  func videoGeometry() {
    let naturalSize = CGSize(width: 1_920, height: 1_080)
    let portrait = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 1_080, ty: 0)
    let sourceSize = VideoUpscalingGeometry.displayPixelSize(
      naturalSize: naturalSize,
      preferredTransform: portrait
    )
    #expect(sourceSize == UpscalingPixelSize(width: 1_080, height: 1_920))

    let targetSize = UpscalingPixelSize(width: 2_160, height: 3_840)
    let transform = VideoUpscalingGeometry.renderTransform(
      naturalSize: naturalSize,
      preferredTransform: portrait,
      targetSize: targetSize
    )
    let rendered = CGRect(origin: .zero, size: naturalSize).applying(transform)
    #expect(abs(rendered.minX) < 0.001)
    #expect(abs(rendered.minY) < 0.001)
    #expect(abs(rendered.width - 2_160) < 0.001)
    #expect(abs(rendered.height - 3_840) < 0.001)

    let encoded = VideoUpscalingGeometry.encodedPixelSize(
      naturalSize: naturalSize,
      preferredTransform: portrait,
      targetDisplaySize: targetSize
    )
    #expect(encoded == UpscalingPixelSize(width: 3_840, height: 2_160))

    let outputOrientation = VideoUpscalingGeometry.outputPreferredTransform(
      encodedPixelSize: encoded,
      sourcePreferredTransform: portrait
    )
    let outputDisplayRect = CGRect(
      x: 0,
      y: 0,
      width: encoded.width,
      height: encoded.height
    ).applying(outputOrientation)
    #expect(abs(outputDisplayRect.minX) < 0.001)
    #expect(abs(outputDisplayRect.minY) < 0.001)
    #expect(abs(outputDisplayRect.width - 2_160) < 0.001)
    #expect(abs(outputDisplayRect.height - 3_840) < 0.001)
  }

  @Test("Video provider advertises temporal processing and its local fallback")
  func videoCapabilities() async {
    let provider = AppleVideoUpscalingProvider { kind, size in
      self.snapshot(kind: kind, size: size, modelStatus: .ready)
    }
    let video = await provider.capabilities(for: request(kind: .video))
    #expect(video.count == 2)
    #expect(video[0].backend == .appleVideoToolbox)
    #expect(video[0].usesTemporalFrames)
    #expect(video[0].supportedMediaKinds == [.video])
    #expect(video[1].backend == .avFoundation)

    let image = await provider.capabilities(for: request(kind: .image))
    #expect(image.isEmpty)
  }

  @Test("Best video mode selects temporal processing only when its model is ready")
  func videoBackendSelection() throws {
    var videoRequest = request(kind: .video)
    videoRequest.mode = .best
    let readyPlan = try AppleVideoUpscalingPlanner.plan(
      for: videoRequest,
      snapshot: snapshot(
        kind: .video,
        size: videoRequest.sourcePixelSize,
        modelStatus: .ready
      )
    )
    #expect(readyPlan.backend == .appleVideoToolbox)
    #expect(readyPlan.videoToolboxScaleFactor == 4)

    let fallbackPlan = try AppleVideoUpscalingPlanner.plan(
      for: videoRequest,
      snapshot: snapshot(
        kind: .video,
        size: videoRequest.sourcePixelSize,
        modelStatus: .downloadRequired
      )
    )
    #expect(fallbackPlan.backend == .avFoundation)
  }

  @Test("Detailed video mode reports a required temporal model download")
  func detailedVideoModeRequiresModel() {
    var videoRequest = request(kind: .video)
    videoRequest.mode = .detailed
    #expect(throws: UpscalingError.modelDownloadRequired) {
      try AppleVideoUpscalingPlanner.plan(
        for: videoRequest,
        snapshot: snapshot(
          kind: .video,
          size: videoRequest.sourcePixelSize,
          modelStatus: .downloadRequired
        )
      )
    }
  }

  @Test("Video provider writes an exact-size local movie")
  func videoProviderWritesOutput() async throws {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-video-upscale-test-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let source = directory.appendingPathComponent("source.mp4")
    let destination = directory.appendingPathComponent("upscaled.mp4")
    try await writeFixtureVideo(to: source, width: 64, height: 48)
    let properties = try await UpscalingVideoProbe.inspect(source)
    let targetSize = UpscalingPixelSize(width: 128, height: 96)
    let request = UpscalingRequest(
      sourceURL: source,
      sourceKind: .video,
      sourcePixelSize: properties.pixelSize,
      sourceDuration: properties.duration,
      targetPixelSize: targetSize,
      destinationURL: destination,
      mode: .fast,
      preservesAudio: true
    )
    var result: UpscalingResult?

    let provider = AppleVideoUpscalingProvider { _, _ in
      Issue.record("Fast video upscaling should not probe the optional model backend")
      return self.snapshot(
        kind: .video,
        size: request.sourcePixelSize,
        modelStatus: .downloadRequired
      )
    }
    for try await event in provider.events(for: request) {
      if case .completed(let completed) = event { result = completed }
    }

    #expect(result?.pixelSize == targetSize)
    let output = try await UpscalingVideoProbe.inspect(destination)
    #expect(output.pixelSize == targetSize)
    #expect(output.duration > 0)
  }

  @Test("Temporal video finalization restores the source audio")
  func temporalVideoFinalizerPreservesAudio() async throws {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-temporal-mux-test-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let source = directory.appendingPathComponent("source-with-audio.mp4")
    let processed = directory.appendingPathComponent("processed.mov")
    let destination = directory.appendingPathComponent("upscaled.mp4")
    try await writeFixtureAudioVideo(to: source, width: 64, height: 48)
    try await writeFixtureVideo(to: processed, width: 128, height: 96, fileType: .mov)
    let sourceProperties = try await UpscalingVideoProbe.inspect(source)
    let request = UpscalingRequest(
      sourceURL: source,
      sourceKind: .video,
      sourcePixelSize: sourceProperties.pixelSize,
      sourceDuration: sourceProperties.duration,
      targetPixelSize: UpscalingPixelSize(width: 128, height: 96),
      destinationURL: destination,
      mode: .detailed,
      preservesAudio: true
    )

    try await AppleTemporalVideoUpscalingPipeline.writeFinalAsset(
      processedVideoURL: processed,
      sourceAsset: AVURLAsset(url: source),
      request: request
    )

    let output = try await UpscalingVideoProbe.inspect(destination)
    #expect(output.pixelSize == request.targetPixelSize)
    #expect(output.hasAudio)
  }

  private func request(kind: MediaKind) -> UpscalingRequest {
    UpscalingRequest(
      sourceURL: URL(fileURLWithPath: "/tmp/source.mov"),
      sourceKind: kind,
      sourcePixelSize: UpscalingPixelSize(width: 640, height: 360),
      sourceDuration: 4,
      targetPixelSize: UpscalingPixelSize(width: 1_280, height: 720),
      destinationURL: URL(fileURLWithPath: "/tmp/output.mov")
    )
  }

  private func snapshot(
    kind: MediaKind,
    size: UpscalingPixelSize,
    modelStatus: UpscalingModelStatus?
  ) -> AppleUpscalingCapabilitySnapshot {
    AppleUpscalingCapabilitySnapshot(
      sourceKind: kind,
      sourcePixelSize: size,
      highQualitySupported: true,
      highQualityScaleFactors: [4],
      highQualityModelStatus: modelStatus,
      lowLatencySupported: true,
      lowLatencyScaleFactors: [2],
      metalFXSupported: false
    )
  }

  private func writeFixtureImage(to url: URL, width: Int, height: Int) throws {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
      throw UpscalingError.failed("The test fixture image could not be created.")
    }
    context.setFillColor(red: 0.2, green: 0.4, blue: 0.8, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    guard let image = context.makeImage(),
      let destination = CGImageDestinationCreateWithURL(
        url as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
      )
    else {
      throw UpscalingError.failed("The test fixture image could not be created.")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw UpscalingError.failed("The test fixture image could not be encoded.")
    }
  }

  private func imageSize(at url: URL) -> UpscalingPixelSize? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
        as? [CFString: Any],
      let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
      let height = properties[kCGImagePropertyPixelHeight] as? NSNumber
    else {
      return nil
    }
    return UpscalingPixelSize(width: width.intValue, height: height.intValue)
  }

  private func writeFixtureVideo(
    to url: URL,
    width: Int,
    height: Int,
    fileType: AVFileType = .mp4
  ) async throws {
    let writer = try AVAssetWriter(outputURL: url, fileType: fileType)
    let input = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
      ]
    )
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: input,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      ]
    )
    guard writer.canAdd(input) else {
      throw UpscalingError.failed("The test video input is unavailable.")
    }
    writer.add(input)
    guard writer.startWriting() else {
      throw writer.error ?? UpscalingError.failed("The test video writer could not start.")
    }
    writer.startSession(atSourceTime: .zero)

    for frame in 0..<2 {
      while !input.isReadyForMoreMediaData { await Task.yield() }
      let pixelBuffer = try fixturePixelBuffer(width: width, height: height)
      let time = CMTime(value: CMTimeValue(frame), timescale: 30)
      guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
        throw writer.error ?? UpscalingError.failed("The test video frame could not be appended.")
      }
    }
    input.markAsFinished()
    await writer.finishWriting()
    guard writer.status == .completed else {
      throw writer.error ?? UpscalingError.failed("The test video could not be finalized.")
    }
  }

  private func writeFixtureAudioVideo(to url: URL, width: Int, height: Int) async throws {
    let directory = url.deletingLastPathComponent()
    let videoURL = directory.appendingPathComponent("\(UUID().uuidString)-video.mp4")
    let audioURL = directory.appendingPathComponent("\(UUID().uuidString)-audio.wav")
    defer {
      try? FileManager.default.removeItem(at: videoURL)
      try? FileManager.default.removeItem(at: audioURL)
    }
    try await writeFixtureVideo(to: videoURL, width: width, height: height)
    try writeFixtureAudio(to: audioURL)

    let videoAsset = AVURLAsset(url: videoURL)
    let audioAsset = AVURLAsset(url: audioURL)
    guard let sourceVideo = try await videoAsset.loadTracks(withMediaType: .video).first,
      let sourceAudio = try await audioAsset.loadTracks(withMediaType: .audio).first
    else {
      throw UpscalingError.failed("The audio-video test tracks could not be read.")
    }
    let duration = try await videoAsset.load(.duration)
    let composition = AVMutableComposition()
    guard let videoTrack = composition.addMutableTrack(
      withMediaType: .video,
      preferredTrackID: kCMPersistentTrackID_Invalid
    ), let audioTrack = composition.addMutableTrack(
      withMediaType: .audio,
      preferredTrackID: kCMPersistentTrackID_Invalid
    ) else {
      throw UpscalingError.failed("The audio-video test composition could not be created.")
    }
    try videoTrack.insertTimeRange(
      CMTimeRange(start: .zero, duration: duration),
      of: sourceVideo,
      at: .zero
    )
    try audioTrack.insertTimeRange(
      CMTimeRange(start: .zero, duration: duration),
      of: sourceAudio,
      at: .zero
    )
    guard let export = AVAssetExportSession(
      asset: composition,
      presetName: AVAssetExportPresetHighestQuality
    ) else {
      throw UpscalingError.failed("The audio-video test exporter could not be created.")
    }
    try await export.export(to: url, as: .mp4)
  }

  private func writeFixtureAudio(to url: URL) throws {
    guard let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1),
      let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4_800),
      let samples = buffer.floatChannelData?[0]
    else {
      throw UpscalingError.failed("The audio test buffer could not be created.")
    }
    buffer.frameLength = buffer.frameCapacity
    for frame in 0..<Int(buffer.frameLength) {
      samples[frame] = sin(Float(frame) * 2 * .pi * 440 / 48_000) * 0.1
    }
    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    try file.write(from: buffer)
  }

  private func fixturePixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      width,
      height,
      kCVPixelFormatType_32BGRA,
      [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary,
      &pixelBuffer
    )
    guard status == kCVReturnSuccess, let pixelBuffer else {
      throw UpscalingError.failed("The test pixel buffer could not be allocated.")
    }
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw UpscalingError.failed("The test pixel buffer could not be accessed.")
    }
    memset(baseAddress, 96, CVPixelBufferGetDataSize(pixelBuffer))
    return pixelBuffer
  }
}
