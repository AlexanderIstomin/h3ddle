import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import H3ddleCore
import ImageIO


public struct ProgramExporter: ProgramExporting, Sendable {
  public init() {}

  public func export(
    project: H3ddleProject,
    settings: ProgramExportSettings,
    destination: URL
  ) -> AsyncThrowingStream<MediaExportEvent, any Error> {
    AsyncThrowingStream { continuation in
      let task = Task.detached(priority: .userInitiated) {
        do {
          try await ExportSessionWriter.write(
            project: project,
            settings: settings,
            destination: destination
          ) { event in
            continuation.yield(event)
          }
          continuation.finish()
        } catch is CancellationError {
          try? FileManager.default.removeItem(at: destination)
          continuation.finish(throwing: MediaExportError.cancelled)
        } catch let error as MediaExportError {
          try? FileManager.default.removeItem(at: destination)
          continuation.finish(throwing: error)
        } catch {
          try? FileManager.default.removeItem(at: destination)
          continuation.finish(throwing: MediaExportError.failed(error.localizedDescription))
        }
      }
      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }
}

private final class AssetWriterBox: @unchecked Sendable {
  let writer: AVAssetWriter
  init(_ writer: AVAssetWriter) { self.writer = writer }
}

private enum ExportSessionWriter {
  static func write(
    project: H3ddleProject,
    settings: ProgramExportSettings,
    destination: URL,
    emit: @Sendable (MediaExportEvent) -> Void
  ) async throws {
    let plan = ProgramCompositionPlan(project: project)
    let span = settings.range.resolved(in: plan.duration)
    let duration = span.outSec - span.inSec
    guard duration > 0.001 else { throw MediaExportError.emptyProgram }

    emit(.preparing)
    try Task.checkCancellation()

    if FileManager.default.fileExists(atPath: destination.path) {
      try FileManager.default.removeItem(at: destination)
    }
    try FileManager.default.createDirectory(
      at: destination.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )

    let size = settings.outputPixelSize(project: project.settings)
    let fps = settings.framesPerSecond
    let frameCount = max(1, Int((duration * fps).rounded(.up)))
    let audioPlan =
      settings.includeAudioLane
      ? try await ExportAudioBuilder.makePlan(project: project, range: span)
      : nil
    let videoURL =
      audioPlan == nil
      ? destination
      : FileManager.default.temporaryDirectory.appendingPathComponent(
        "h3ddle-video-\(UUID().uuidString).\(settings.format.fileExtension)"
      )

    try await writeVideo(
      project: project,
      settings: settings,
      span: span,
      size: size,
      frameCount: frameCount,
      destination: videoURL,
      emit: emit
    )

    if let audioPlan {
      emit(.progress(phase: "Mixing audio", fraction: 0.84))
      try await mux(
        video: videoURL,
        audio: audioPlan,
        settings: settings,
        destination: destination
      )
      try? FileManager.default.removeItem(at: videoURL)
    }

    try Task.checkCancellation()
    emit(.progress(phase: "Finalizing file", fraction: 1))
    emit(.completed(destination))
  }

  private static func writeVideo(
    project: H3ddleProject,
    settings: ProgramExportSettings,
    span: (inSec: TimeInterval, outSec: TimeInterval),
    size: (width: Int, height: Int),
    frameCount: Int,
    destination: URL,
    emit: @Sendable (MediaExportEvent) -> Void
  ) async throws {
    if settings.usesHardwareAcceleration {
      try await writeHardwareVideo(
        project: project,
        settings: settings,
        span: span,
        size: size,
        frameCount: frameCount,
        destination: destination,
        emit: emit
      )
    } else {
      try await writeSoftwareVideo(
        project: project,
        settings: settings,
        span: span,
        size: size,
        frameCount: frameCount,
        destination: destination,
        emit: emit
      )
    }
  }

  private static func writeHardwareVideo(
    project: H3ddleProject,
    settings: ProgramExportSettings,
    span: (inSec: TimeInterval, outSec: TimeInterval),
    size: (width: Int, height: Int),
    frameCount: Int,
    destination: URL,
    emit: @Sendable (MediaExportEvent) -> Void
  ) async throws {
    let writer = try AVAssetWriter(outputURL: destination, fileType: settings.format.fileType)
    let videoInput = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: settings.videoOutputSettings(width: size.width, height: size.height)
    )
    videoInput.expectsMediaDataInRealTime = false
    guard writer.canAdd(videoInput) else {
      throw MediaExportError.failed("The video writer rejected the encode settings.")
    }
    writer.add(videoInput)
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: videoInput,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: size.width,
        kCVPixelBufferHeightKey as String: size.height,
      ]
    )
    guard writer.startWriting() else {
      throw MediaExportError.failed(writer.error?.localizedDescription ?? "Could not start the writer.")
    }
    writer.startSession(atSourceTime: .zero)

    let renderer = ProgramCompositor(
      width: size.width,
      height: size.height,
      background: project.settings.background.compositorColor,
      layoutWidth: project.settings.width,
      layoutHeight: project.settings.height
    )
    let timescale = CMTimeScale(max(1, settings.framesPerSecond.rounded()))
    var lastPreview = ContinuousClock.now - .seconds(1)
    for index in 0..<frameCount {
      try Task.checkCancellation()
      try await waitUntilReady(videoInput)
      guard let buffer = await renderer.pixelBuffer(for: frame(at: index, span: span, settings: settings, project: project))
      else {
        throw MediaExportError.failed("Could not render frame \(index).")
      }
      emitPreview(buffer, index: index, frameCount: frameCount, lastPreview: &lastPreview, emit: emit)
      let presentation = CMTime(value: CMTimeValue(index), timescale: timescale)
      if !adaptor.append(buffer, withPresentationTime: presentation) {
        throw MediaExportError.failed(
          writer.error?.localizedDescription ?? "Could not append video frame \(index)."
        )
      }
      emit(.progress(phase: "Encoding video", fraction: Double(index + 1) / Double(frameCount) * 0.82))
    }
    videoInput.markAsFinished()
    emit(.progress(phase: "Finalizing file", fraction: 0.96))
    try await finish(writer)
  }

  private static func writeSoftwareVideo(
    project: H3ddleProject,
    settings: ProgramExportSettings,
    span: (inSec: TimeInterval, outSec: TimeInterval),
    size: (width: Int, height: Int),
    frameCount: Int,
    destination: URL,
    emit: @Sendable (MediaExportEvent) -> Void
  ) async throws {
    let encoder = try SoftwareVideoEncoder(
      width: size.width,
      height: size.height,
      settings: settings
    )
    let renderer = ProgramCompositor(
      width: size.width,
      height: size.height,
      background: project.settings.background.compositorColor,
      layoutWidth: project.settings.width,
      layoutHeight: project.settings.height
    )
    let timescale = CMTimeScale(max(1, settings.framesPerSecond.rounded()))
    let frameDuration = CMTime(value: 1, timescale: timescale)
    var lastPreview = ContinuousClock.now - .seconds(1)
    for index in 0..<frameCount {
      try Task.checkCancellation()
      guard let buffer = await renderer.pixelBuffer(for: frame(at: index, span: span, settings: settings, project: project))
      else {
        throw MediaExportError.failed("Could not render frame \(index).")
      }
      emitPreview(buffer, index: index, frameCount: frameCount, lastPreview: &lastPreview, emit: emit)
      try encoder.encode(
        buffer,
        at: CMTime(value: CMTimeValue(index), timescale: timescale),
        duration: frameDuration
      )
      emit(.progress(phase: "Encoding video", fraction: Double(index + 1) / Double(frameCount) * 0.72))
    }
    let samples = try encoder.finish()
    try await writePassthroughVideo(samples, settings: settings, destination: destination)
    emit(.progress(phase: "Finalizing file", fraction: 0.82))
  }

  private static func writePassthroughVideo(
    _ samples: [CMSampleBuffer],
    settings: ProgramExportSettings,
    destination: URL
  ) async throws {
    guard let first = samples.first,
      let hint = CMSampleBufferGetFormatDescription(first)
    else {
      throw MediaExportError.failed("Software encoder produced an unreadable frame.")
    }
    let writer = try AVAssetWriter(outputURL: destination, fileType: settings.format.fileType)
    let videoInput = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: nil,
      sourceFormatHint: hint
    )
    videoInput.expectsMediaDataInRealTime = false
    guard writer.canAdd(videoInput) else {
      throw MediaExportError.failed("The video writer rejected the software bitstream.")
    }
    writer.add(videoInput)
    guard writer.startWriting() else {
      throw MediaExportError.failed(writer.error?.localizedDescription ?? "Could not start the writer.")
    }
    writer.startSession(atSourceTime: .zero)
    for sample in samples {
      try await waitUntilReady(videoInput)
      if !videoInput.append(sample) {
        throw MediaExportError.failed("Could not write a software-encoded frame.")
      }
    }
    videoInput.markAsFinished()
    try await finish(writer)
  }

  private static func mux(
    video: URL,
    audio: ExportAudioPlan,
    settings: ProgramExportSettings,
    destination: URL
  ) async throws {
    let audioURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3ddle-audio-\(UUID().uuidString).m4a")
    defer { try? FileManager.default.removeItem(at: audioURL) }
    try await writeAudioFile(plan: audio, settings: settings, to: audioURL)

    if FileManager.default.fileExists(atPath: destination.path) {
      try FileManager.default.removeItem(at: destination)
    }

    let composition = AVMutableComposition()
    try await insertTracks(
      from: AVURLAsset(url: video),
      mediaType: .video,
      into: composition
    )
    try await insertTracks(
      from: AVURLAsset(url: audioURL),
      mediaType: .audio,
      into: composition
    )

    guard
      let session = AVAssetExportSession(
        asset: composition,
        presetName: AVAssetExportPresetPassthrough
      )
    else {
      throw MediaExportError.failed("Could not start the audio mix.")
    }
    do {
      try await session.export(to: destination, as: settings.format.fileType)
    } catch is CancellationError {
      throw MediaExportError.cancelled
    } catch {
      throw MediaExportError.failed(error.localizedDescription)
    }
  }

  private static func writeAudioFile(
    plan: ExportAudioPlan,
    settings: ProgramExportSettings,
    to url: URL
  ) async throws {
    if FileManager.default.fileExists(atPath: url.path) {
      try FileManager.default.removeItem(at: url)
    }
    let writer = try AVAssetWriter(outputURL: url, fileType: .m4a)
    let audioInput = AVAssetWriterInput(
      mediaType: .audio,
      outputSettings: settings.audioOutputSettings()
    )
    audioInput.expectsMediaDataInRealTime = false
    guard writer.canAdd(audioInput) else {
      throw MediaExportError.failed("The audio writer rejected the encode settings.")
    }
    writer.add(audioInput)
    guard writer.startWriting() else {
      throw MediaExportError.failed(
        writer.error?.localizedDescription ?? "Could not start the audio writer."
      )
    }
    writer.startSession(atSourceTime: .zero)
    try await ExportAudioBuilder.write(
      plan: plan,
      to: audioInput,
      normalize: settings.normalizeLoudness
    )
    audioInput.markAsFinished()
    try await finish(writer)
  }

  private static func insertTracks(
    from asset: AVAsset,
    mediaType: AVMediaType,
    into composition: AVMutableComposition
  ) async throws {
    let tracks = try await asset.loadTracks(withMediaType: mediaType)
    for track in tracks {
      guard
        let destination = composition.addMutableTrack(
          withMediaType: mediaType,
          preferredTrackID: kCMPersistentTrackID_Invalid
        )
      else {
        continue
      }
      let timeRange = try await track.load(.timeRange)
      try destination.insertTimeRange(timeRange, of: track, at: timeRange.start)
    }
  }

  private static func emitPreview(
    _ buffer: CVPixelBuffer,
    index: Int,
    frameCount: Int,
    lastPreview: inout ContinuousClock.Instant,
    emit: @Sendable (MediaExportEvent) -> Void
  ) {
    let now = ContinuousClock.now
    let shouldEmit =
      index == 0
      || index == frameCount - 1
      || now - lastPreview >= .milliseconds(66)
    guard shouldEmit, let image = ProgramCompositor.makeImage(from: buffer) else { return }
    lastPreview = now
    emit(.preview(ExportPreviewImage(image: image)))
  }

  private static func frame(
    at index: Int,
    span: (inSec: TimeInterval, outSec: TimeInterval),
    settings: ProgramExportSettings,
    project: H3ddleProject
  ) -> ProgramPreviewFrame {
    let programTime = min(
      span.outSec - 0.000_1,
      span.inSec + (Double(index) + 0.5) / settings.framesPerSecond
    )
    return ProgramPreview.frame(
      at: programTime,
      project: project,
      textMuted: !settings.includeTextLane
    )
  }

  private static func waitUntilReady(_ input: AVAssetWriterInput) async throws {
    let deadline = ContinuousClock.now + .seconds(20)
    while !input.isReadyForMoreMediaData {
      try Task.checkCancellation()
      if ContinuousClock.now > deadline {
        throw MediaExportError.failed("The encoder stopped accepting media.")
      }
      try await Task.sleep(for: .milliseconds(4))
    }
  }

  private static func finish(_ writer: AVAssetWriter) async throws {
    let box = AssetWriterBox(writer)
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
      box.writer.finishWriting {
        if box.writer.status == .completed {
          continuation.resume()
        } else {
          continuation.resume(
            throwing: MediaExportError.failed(
              box.writer.error?.localizedDescription ?? "The writer could not finish the file."
            )
          )
        }
      }
    }
  }
}

private struct ExportAudioPlan {
  var composition: AVMutableComposition
  var trackVolumes: [(AVMutableCompositionTrack, Float)]

  func mix(gain: Float = 1) -> AVMutableAudioMix {
    let mix = AVMutableAudioMix()
    mix.inputParameters = trackVolumes.map { track, volume in
      let parameters = AVMutableAudioMixInputParameters(track: track)
      parameters.setVolume(max(0, volume * gain), at: .zero)
      return parameters
    }
    return mix
  }
}

private enum ExportAudioBuilder {
  static func makePlan(
    project: H3ddleProject,
    range: (inSec: TimeInterval, outSec: TimeInterval)
  ) async throws -> ExportAudioPlan? {
    let composition = AVMutableComposition()
    var trackVolumes: [(AVMutableCompositionTrack, Float)] = []
    let master = Float(project.settings.masterGain)

    for item in project.timeline.audioItems where item.isEnabled {
      guard let asset = project.asset(id: item.assetID) else { continue }
      try await append(
        url: asset.url,
        sourceStart: item.sourceOffset,
        sourceDuration: item.duration,
        programStart: item.startTime,
        volume: item.gain * master,
        range: range,
        into: composition,
        trackVolumes: &trackVolumes
      )
    }

    for placement in project.timeline.visualPlacements {
      let item = placement.item
      guard item.isEnabled, item.includesNativeAudio else { continue }
      guard let asset = project.asset(id: item.assetID), asset.kind == .video else { continue }
      try await append(
        url: asset.url,
        sourceStart: item.sourceOffset,
        sourceDuration: item.duration,
        programStart: placement.startTime,
        volume: master,
        range: range,
        into: composition,
        trackVolumes: &trackVolumes
      )
    }

    let tracks = try await composition.loadTracks(withMediaType: .audio)
    guard !tracks.isEmpty else { return nil }
    return ExportAudioPlan(composition: composition, trackVolumes: trackVolumes)
  }

  static func write(
    plan: ExportAudioPlan,
    to input: AVAssetWriterInput,
    normalize: Bool
  ) async throws {
    let sampleRate = 48_000.0
    let channels = 2
    let pcmSettings: [String: Any] = [
      AVFormatIDKey: kAudioFormatLinearPCM,
      AVLinearPCMBitDepthKey: 32,
      AVLinearPCMIsFloatKey: true,
      AVLinearPCMIsNonInterleaved: false,
      AVSampleRateKey: sampleRate,
      AVNumberOfChannelsKey: channels,
    ]

    let mix: AVMutableAudioMix
    if normalize {
      let samples = try await readSamples(plan: plan, settings: pcmSettings, mix: plan.mix())
      let measured = Loudness.integratedLUFS(
        samples: samples,
        sampleRate: sampleRate,
        channels: channels
      )
      mix = plan.mix(gain: Loudness.gain(from: measured))
    } else {
      mix = plan.mix()
    }
    try await copySamples(
      try await readSampleBuffers(plan: plan, settings: pcmSettings, mix: mix),
      to: input
    )
  }

  private static func append(
    url: URL,
    sourceStart: TimeInterval,
    sourceDuration: TimeInterval,
    programStart: TimeInterval,
    volume: Float,
    range: (inSec: TimeInterval, outSec: TimeInterval),
    into composition: AVMutableComposition,
    trackVolumes: inout [(AVMutableCompositionTrack, Float)]
  ) async throws {
    let overlapStart = max(programStart, range.inSec)
    let overlapEnd = min(programStart + sourceDuration, range.outSec)
    guard overlapEnd > overlapStart + 0.001 else { return }
    guard FileManager.default.fileExists(atPath: url.path) else { return }

    let asset = AVURLAsset(url: url)
    let tracks = try await asset.loadTracks(withMediaType: .audio)
    guard let sourceTrack = tracks.first else { return }
    guard
      let destination = composition.addMutableTrack(
        withMediaType: .audio,
        preferredTrackID: kCMPersistentTrackID_Invalid
      )
    else {
      return
    }

    let localIn = sourceStart + (overlapStart - programStart)
    let destStart = overlapStart - range.inSec
    try destination.insertTimeRange(
      CMTimeRange(
        start: CMTime(seconds: localIn, preferredTimescale: 600),
        duration: CMTime(seconds: overlapEnd - overlapStart, preferredTimescale: 600)
      ),
      of: sourceTrack,
      at: CMTime(seconds: destStart, preferredTimescale: 600)
    )
    trackVolumes.append((destination, max(0, volume)))
  }

  private static func readSamples(
    plan: ExportAudioPlan,
    settings: [String: Any],
    mix: AVAudioMix
  ) async throws -> [Float] {
    var samples: [Float] = []
    for buffer in try await readSampleBuffers(plan: plan, settings: settings, mix: mix) {
      appendFloats(from: buffer, into: &samples)
    }
    return samples
  }

  private static func readSampleBuffers(
    plan: ExportAudioPlan,
    settings: [String: Any],
    mix: AVAudioMix
  ) async throws -> [CMSampleBuffer] {
    let reader = try AVAssetReader(asset: plan.composition)
    let tracks = try await plan.composition.loadTracks(withMediaType: .audio)
    let output = AVAssetReaderAudioMixOutput(audioTracks: tracks, audioSettings: settings)
    output.audioMix = mix
    guard reader.canAdd(output) else {
      throw MediaExportError.failed("Could not prepare the audio mix.")
    }
    reader.add(output)
    guard reader.startReading() else {
      throw MediaExportError.failed(
        reader.error?.localizedDescription ?? "Could not read mixed audio."
      )
    }

    var buffers: [CMSampleBuffer] = []
    while reader.status == .reading {
      try Task.checkCancellation()
      if let buffer = output.copyNextSampleBuffer() {
        buffers.append(buffer)
      } else {
        break
      }
    }
    if reader.status == .failed {
      throw MediaExportError.failed(
        reader.error?.localizedDescription ?? "Audio mix reading failed."
      )
    }
    return buffers
  }

  private static func copySamples(
    _ buffers: [CMSampleBuffer],
    to input: AVAssetWriterInput
  ) async throws {
    for buffer in buffers {
      try Task.checkCancellation()
      while !input.isReadyForMoreMediaData {
        try Task.checkCancellation()
        try await Task.sleep(for: .milliseconds(4))
      }
      if !input.append(buffer) {
        throw MediaExportError.failed("Could not append mixed audio.")
      }
    }
  }

  private static func appendFloats(from buffer: CMSampleBuffer, into samples: inout [Float]) {
    guard let block = CMSampleBufferGetDataBuffer(buffer) else { return }
    var length = 0
    var dataPointer: UnsafeMutablePointer<Int8>?
    CMBlockBufferGetDataPointer(
      block,
      atOffset: 0,
      lengthAtOffsetOut: nil,
      totalLengthOut: &length,
      dataPointerOut: &dataPointer
    )
    guard let dataPointer else { return }
    let count = length / MemoryLayout<Float>.size
    dataPointer.withMemoryRebound(to: Float.self, capacity: count) { pointer in
      samples.append(contentsOf: UnsafeBufferPointer(start: pointer, count: count))
    }
  }
}

extension ProgramExportFormat {
  fileprivate var fileType: AVFileType {
    switch self {
    case .h264, .h265: .mp4
    case .proRes: .mov
    }
  }

  fileprivate var codec: AVVideoCodecType {
    switch self {
    case .h264: .h264
    case .h265: .hevc
    case .proRes: .proRes422
    }
  }
}

extension ProgramExportSettings {
  fileprivate func videoOutputSettings(width: Int, height: Int) -> [String: Any] {
    var compression: [String: Any] = [
      AVVideoExpectedSourceFrameRateKey: framesPerSecond
    ]
    switch format {
    case .h264:
      compression[AVVideoAverageBitRateKey] = Int(videoBitrateKbps * 1_000)
      if let level = profile.avFoundationLevel {
        compression[AVVideoProfileLevelKey] = level
      }
    case .h265:
      compression[AVVideoAverageBitRateKey] = Int(videoBitrateKbps * 1_000)
    case .proRes:
      break
    }
    return [
      AVVideoCodecKey: format.codec,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoCompressionPropertiesKey: compression,
    ]
  }

  fileprivate func audioOutputSettings() -> [String: Any] {
    [
      AVFormatIDKey: kAudioFormatMPEG4AAC,
      AVSampleRateKey: 48_000,
      AVNumberOfChannelsKey: 2,
      AVEncoderBitRateKey: audioBitrateKbps * 1_000,
    ]
  }
}

extension ProgramExportProfile {
  fileprivate var avFoundationLevel: String? {
    switch self {
    case .auto: nil
    case .baseline: AVVideoProfileLevelH264BaselineAutoLevel
    case .main: AVVideoProfileLevelH264MainAutoLevel
    case .high: AVVideoProfileLevelH264HighAutoLevel
    }
  }
}

