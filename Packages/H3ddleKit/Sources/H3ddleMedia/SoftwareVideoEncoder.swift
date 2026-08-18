import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

/// Encodes pixel buffers with a VideoToolbox session so hardware can be refused
/// without passing unsupported keys to AVAssetWriter.
final class SoftwareVideoEncoder: @unchecked Sendable {
  private let session: VTCompressionSession
  private let sink = SampleSink()

  init(width: Int, height: Int, settings: ProgramExportSettings) throws {
    let spec: [String: Any] = [
      kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder as String: false
    ]
    var session: VTCompressionSession?
    let status = VTCompressionSessionCreate(
      allocator: nil,
      width: Int32(width),
      height: Int32(height),
      codecType: settings.format.coreMediaCodecType,
      encoderSpecification: spec as CFDictionary,
      imageBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey: width,
        kCVPixelBufferHeightKey: height,
      ] as CFDictionary,
      compressedDataAllocator: nil,
      outputCallback: softwareEncoderCallback,
      refcon: Unmanaged.passUnretained(sink).toOpaque(),
      compressionSessionOut: &session
    )
    guard status == noErr, let session else {
      throw MediaExportError.failed("Could not start a software video encoder (\(status)).")
    }
    self.session = session

    VTSessionSetProperty(session, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanFalse)
    VTSessionSetProperty(
      session,
      key: kVTCompressionPropertyKey_AllowFrameReordering,
      value: kCFBooleanFalse
    )
    VTSessionSetProperty(
      session,
      key: kVTCompressionPropertyKey_ExpectedFrameRate,
      value: settings.framesPerSecond as CFNumber
    )
    if settings.format != .proRes {
      VTSessionSetProperty(
        session,
        key: kVTCompressionPropertyKey_AverageBitRate,
        value: Int(settings.videoBitrateKbps * 1_000) as CFNumber
      )
    }
    if let profile = settings.profile.coreMediaProfile, settings.format == .h264 {
      VTSessionSetProperty(session, key: kVTCompressionPropertyKey_ProfileLevel, value: profile)
    }
  }

  func encode(_ buffer: CVPixelBuffer, at presentation: CMTime, duration: CMTime) throws {
    try Task.checkCancellation()
    let status = VTCompressionSessionEncodeFrame(
      session,
      imageBuffer: buffer,
      presentationTimeStamp: presentation,
      duration: duration,
      frameProperties: nil,
      sourceFrameRefcon: nil,
      infoFlagsOut: nil
    )
    if status != noErr {
      throw MediaExportError.failed("Software encoder rejected a frame (\(status)).")
    }
    try sink.thrownError()
  }

  func completeFrames(through presentation: CMTime) throws -> [CMSampleBuffer] {
    try completeFrames(until: presentation)
  }

  func finish() throws -> [CMSampleBuffer] {
    try completeFrames(until: .invalid)
  }

  private func completeFrames(until presentation: CMTime) throws -> [CMSampleBuffer] {
    let status = VTCompressionSessionCompleteFrames(
      session,
      untilPresentationTimeStamp: presentation
    )
    if status != noErr {
      throw MediaExportError.failed("Software encoder could not finish (\(status)).")
    }
    try sink.thrownError()
    return sink.takeEncodedSamples()
  }

  deinit {
    VTCompressionSessionInvalidate(session)
  }
}

private final class SampleSink: @unchecked Sendable {
  private let lock = NSLock()
  private var samples: [CMSampleBuffer] = []
  private var encodeError: Error?

  func handle(status: OSStatus, sample: CMSampleBuffer?) {
    lock.lock()
    defer { lock.unlock() }
    if status != noErr {
      encodeError = MediaExportError.failed("Software encoder failed (\(status)).")
      return
    }
    if let sample {
      samples.append(sample)
    }
  }

  func takeEncodedSamples() -> [CMSampleBuffer] {
    lock.lock()
    defer { lock.unlock() }
    let encoded = samples
    samples.removeAll(keepingCapacity: true)
    return encoded
  }

  func thrownError() throws {
    lock.lock()
    let error = encodeError
    lock.unlock()
    if let error { throw error }
  }
}

private let softwareEncoderCallback: VTCompressionOutputCallback = { refcon, _, status, _, sample in
  guard let refcon else { return }
  Unmanaged<SampleSink>.fromOpaque(refcon).takeUnretainedValue()
    .handle(status: status, sample: sample)
}

extension ProgramExportFormat {
  fileprivate var coreMediaCodecType: CMVideoCodecType {
    switch self {
    case .h264: kCMVideoCodecType_H264
    case .h265: kCMVideoCodecType_HEVC
    case .proRes: kCMVideoCodecType_AppleProRes422
    }
  }
}

extension ProgramExportProfile {
  fileprivate var coreMediaProfile: CFString? {
    switch self {
    case .auto: nil
    case .baseline: kVTProfileLevel_H264_Baseline_AutoLevel
    case .main: kVTProfileLevel_H264_Main_AutoLevel
    case .high: kVTProfileLevel_H264_High_AutoLevel
    }
  }
}
