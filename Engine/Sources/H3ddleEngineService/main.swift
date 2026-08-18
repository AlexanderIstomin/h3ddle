import CoreGraphics
import Foundation
import H3Native
import H3ddleEngineProtocol
import ImageIO
import UniformTypeIdentifiers

@main
enum H3ddleEngineService {
  static func main() {
    EngineResourceWatch.shared.start()
    while let line = readLine() {
      guard
        let command = try? EngineLineCodec.decode(
          EngineCommand.self,
          from: Data(line.utf8)
        )
      else {
        continue
      }

      guard command.protocolVersion == H3ddleEngineProtocol.currentVersion else {
        EngineOutput.emit(
          EngineEvent(
            requestID: command.requestID,
            jobID: command.jobID,
            kind: .failed,
            message: "Unsupported engine protocol version"
          )
        )
        continue
      }

      switch command.kind {
      case .handshake:
        EngineResourceWatch.shared.noteActivity()
        prepareMetal()
        EngineOutput.emit(
          EngineEvent(
            requestID: command.requestID,
            kind: .ready,
            capabilities: engineCapabilities,
            message: "H3 native engine ready"
          )
        )
      case .inspectModel:
        EngineResourceWatch.shared.noteActivity()
        inspectModel(for: command)
      case .generate:
        EngineResourceWatch.shared.noteActivity()
        EngineRuntime.shared.start(command)
      case .cancel:
        EngineRuntime.shared.cancel(command)
      case .shutdown:
        EngineRuntime.shared.cancelActiveJob()
        EngineModelStore.shared.release()
        return
      }
    }
  }

  private static func inspectModel(for command: EngineCommand) {
    guard !EngineRuntime.shared.isBusy else {
      EngineOutput.fail(command, message: "Model inspection is unavailable during generation")
      return
    }
    guard let request = command.modelInspection else {
      EngineOutput.fail(command, message: "Model inspection parameters are required")
      return
    }
    guard request.modelDirectory.isFileURL else {
      EngineOutput.fail(command, message: "The model location must be a local directory")
      return
    }

    request.modelDirectory.path.withCString { modelPath in
      guard let context = EngineModelStore.shared.load(path: modelPath) else {
        EngineOutput.fail(command, message: lastH3Error(nil))
        return
      }

      guard let model = h3_model(context), let device = h3_device(context) else {
        EngineOutput.fail(command, message: "H3 did not return model metadata")
        return
      }

      EngineOutput.emit(
        EngineEvent(
          requestID: command.requestID,
          kind: .modelInspected,
          capabilities: engineCapabilities,
          model: EngineModelReport(
            modelDirectory: request.modelDirectory,
            components: modelComponents(model.pointee),
            device: deviceReport(device.pointee),
            format: modelFormat(model.pointee),
            supportsGeneration: h3ddle_h3_model_supports_generation(model) != 0
          )
        )
      )
    }
  }
}

private let engineCapabilities = EngineCapabilities(
  engineName: "h3.c",
  engineVersion: String(cString: h3ddle_h3_version()),
  features: {
    // Nothing here depends on an installed FFmpeg any more: audio writes its
    // own WAV, and video muxes through AVFoundation, which ships with macOS.
    // FFmpeg remains a fallback inside the engine for media it reads better,
    // so its absence no longer costs a capability.
    [
      .modelInspection, .videoGeneration, .imageGeneration, .embeddedAudio,
      .cancellation, .denoisingPreviews, .referenceInputs,
      .standaloneAudioGeneration, .soundEffectGeneration, .speechGeneration,
      .zImageGeneration, .ltxGeneration,
    ]
  }()
)

private func prepareMetal() {
  var error = [CChar](repeating: 0, count: 512)
  _ = h3ddle_h3_prepare_metal("h3_shaders.metal", &error, error.count)
}

/// One loaded `h3_ctx` for the helper lifetime. Cache stays enabled so a
/// later generate can reuse exact BF16 conditioning, the prepared DiT, and
/// the video decoder. Idle time and memory pressure drop the cache while
/// leaving the process (and compiled Metal) in place.
private final class EngineModelStore: @unchecked Sendable {
  static let shared = EngineModelStore()

  private let lock = NSLock()
  private var context: OpaquePointer?
  private var directory: String?

  /// Directory of the weights currently in memory, or nil when the cache
  /// has been dropped by idle time or memory pressure.
  var residentDirectory: String? {
    lock.lock()
    defer { lock.unlock() }
    return context == nil ? nil : directory
  }

  func load(path: UnsafePointer<CChar>) -> OpaquePointer? {
    let directoryPath = String(cString: path)
    lock.lock()
    defer { lock.unlock() }
    if let context, directory == directoryPath {
      return context
    }
    releaseLocked()
    // Tens of gigabytes are about to be mapped; anything the audio models
    // are holding should go first.
    h3ddle_sa3_release()
    h3ddle_qwen_release()
    guard let loaded = h3_load_dir(path) else { return nil }
    h3_cache_set_enabled(loaded, 1)
    context = loaded
    directory = directoryPath
    return loaded
  }

  func release() {
    lock.lock()
    defer { lock.unlock() }
    releaseLocked()
  }

  func clearCache() {
    lock.lock()
    defer { lock.unlock() }
    if let context {
      h3_cache_clear(context)
    }
  }

  private func releaseLocked() {
    if let context {
      h3_free(context)
    }
    context = nil
    directory = nil
  }
}

/// Drops retained DiT/VAE/conditioning when the helper is idle or the OS
/// asks for memory. Generation takes the runtime lock first, so eviction
/// cannot free a context that is mid-forward.
private final class EngineResourceWatch: @unchecked Sendable {
  static let shared = EngineResourceWatch()

  private let queue = DispatchQueue(label: "h3ddle.engine.resources")
  private var idleTimer: DispatchSourceTimer?
  private var memorySource: DispatchSourceMemoryPressure?
  private var started = false

  func start() {
    queue.sync {
      guard !started else { return }
      started = true
      let memory = DispatchSource.makeMemoryPressureSource(
        eventMask: [.warning, .critical],
        queue: queue
      )
      memory.setEventHandler { [weak self] in
        self?.evict(releasingContext: true)
      }
      memory.resume()
      memorySource = memory
      armIdleTimerLocked()
    }
  }

  func noteActivity() {
    queue.async { [weak self] in
      self?.armIdleTimerLocked()
    }
  }

  private func armIdleTimerLocked() {
    idleTimer?.cancel()
    let timer = DispatchSource.makeTimerSource(queue: queue)
    timer.schedule(deadline: .now() + .seconds(600), repeating: .never)
    timer.setEventHandler { [weak self] in
      self?.evict(releasingContext: false)
    }
    timer.resume()
    idleTimer = timer
  }

  private func evict(releasingContext: Bool) {
    _ = EngineRuntime.shared.performIfIdle {
      if releasingContext {
        EngineModelStore.shared.release()
      } else {
        EngineModelStore.shared.clearCache()
      }
      // The audio packages go either way. They are held only to save a
      // reload between takes, and after ten idle minutes — or once the OS
      // has asked for memory — there is no take to save.
      h3ddle_sa3_release()
      h3ddle_qwen_release()
    }
  }
}

private func resolveTool(named name: String, override key: String) -> String? {
  let environment = ProcessInfo.processInfo.environment
  if let override = environment[key], FileManager.default.isExecutableFile(atPath: override) {
    return override
  }
  return (environment["PATH"] ?? "")
    .split(separator: ":")
    .map { URL(fileURLWithPath: String($0)).appendingPathComponent(name).path }
    .first(where: FileManager.default.isExecutableFile(atPath:))
}

private func executableIsAvailable(named name: String, override key: String) -> Bool {
  resolveTool(named: name, override: key) != nil
}

private func withOptionalCString(
  _ value: String?,
  _ body: (UnsafePointer<CChar>?) -> Void
) {
  if let value {
    value.withCString(body)
  } else {
    body(nil)
  }
}

private func withCStringList(
  _ values: [String],
  _ body: ([UnsafePointer<CChar>]) -> Void
) {
  func step(_ index: Int, _ collected: [UnsafePointer<CChar>]) {
    if index == values.count {
      body(collected)
      return
    }
    values[index].withCString { pointer in
      step(index + 1, collected + [pointer])
    }
  }
  step(0, [])
}

private func modelComponents(_ model: h3_model_info) -> [EngineModelComponent] {
  [
    component(.textEncoder, model.text_encoder),
    component(.videoTransformer, model.fl2va_transformer),
    component(.referenceTransformer, model.ref2va_transformer),
    component(.videoVAE, model.video_vae),
    component(.audioVAE, model.audio_vae),
  ]
}

private func component(
  _ kind: EngineModelComponentKind,
  _ info: h3_component_info
) -> EngineModelComponent {
  EngineModelComponent(
    kind: kind,
    bytes: info.bytes,
    tensorBytes: info.tensor_bytes,
    fileCount: Int(info.files),
    tensorCount: Int(info.tensors)
  )
}

private func deviceReport(_ device: h3_device_info) -> EngineDeviceReport {
  withUnsafePointer(to: device) { pointer in
    EngineDeviceReport(
      name: String(cString: h3ddle_h3_device_name(pointer)),
      architecture: String(cString: h3ddle_h3_device_architecture(pointer)),
      physicalMemory: device.physical_memory,
      recommendedWorkingSet: device.recommended_working_set,
      unifiedMemory: device.unified_memory != 0
    )
  }
}

private func modelFormat(_ model: h3_model_info) -> EngineModelFormat {
  withUnsafePointer(to: model) { pointer in
    EngineModelFormat(
      rawValue: String(cString: h3ddle_h3_model_layout_name(pointer))
    ) ?? .unknown
  }
}

private func lastH3Error(_ context: OpaquePointer?) -> String {
  guard let message = h3_last_error(context), !String(cString: message).isEmpty else {
    return "The H3 model could not be loaded"
  }
  return String(cString: message)
}

private enum EngineOutput {
  private static let lock = NSLock()

  static func emit(_ event: EngineEvent) {
    var event = event
    if event.residency == nil {
      event.residency = EngineResidency(
        videoModelDirectory: EngineModelStore.shared.residentDirectory.map {
          URL(fileURLWithPath: $0)
        }
      )
    }
    guard let data = try? EngineLineCodec.encode(event) else { return }
    lock.withLock {
      FileHandle.standardOutput.write(data)
    }
  }

  static func fail(_ command: EngineCommand, message: String) {
    emit(
      EngineEvent(
        requestID: command.requestID,
        jobID: command.jobID,
        kind: .failed,
        message: message
      )
    )
  }
}

/// A raw pointer to this object crosses the C callback boundary. Its mutable
/// state is lock-protected, and the generation thread retains it until H3
/// returns; this is the narrow Sendable exception required by the C API.
private final class GenerationCallbackContext: @unchecked Sendable {
  let requestID: UUID
  let jobID: UUID
  let previewURL: URL
  let capturesStill: Bool

  private let lock = NSLock()
  private var cancelled = false
  private var stillPNG: Data?

  init(requestID: UUID, jobID: UUID, previewURL: URL, capturesStill: Bool) {
    self.requestID = requestID
    self.jobID = jobID
    self.previewURL = previewURL
    self.capturesStill = capturesStill
  }

  var isCancelled: Bool {
    lock.withLock { cancelled }
  }

  func cancel() {
    lock.withLock { cancelled = true }
  }

  func progress(phase: String, completed: Int32, total: Int32) -> Int32 {
    let shouldCancel = isCancelled
    let fraction = total > 0 ? Double(completed) / Double(total) : 0
    EngineOutput.emit(
      EngineEvent(
        requestID: requestID,
        jobID: jobID,
        kind: .progress,
        phase: phase,
        fractionComplete: fraction
      )
    )
    return shouldCancel ? 1 : 0
  }

  /// Writes an intermediate denoising frame and announces it. A failed
  /// preview must never stop the generation, so encoding problems are
  /// swallowed; only user cancellation reports back to H3.
  func captureStill(_ frame: h3_frame) -> Int32 {
    if isCancelled { return 1 }
    if let png = Self.encodePNG(frame) {
      lock.withLock { stillPNG = png }
    }
    return isCancelled ? 1 : 0
  }

  func writeStill(to url: URL) throws {
    guard let stillPNG = lock.withLock({ stillPNG }) else {
      throw EngineStillError.missingFrame
    }
    try stillPNG.write(to: url, options: .atomic)
  }

  func deliverPreview(_ frame: h3_frame) -> Int32 {
    if isCancelled { return 1 }
    if let png = Self.encodePNG(frame) {
      do {
        try png.write(to: previewURL, options: .atomic)
        EngineOutput.emit(
          EngineEvent(
            requestID: requestID,
            jobID: jobID,
            kind: .preview,
            phase: "denoise",
            fractionComplete: frame.denoise_steps > 0
              ? Double(frame.denoise_step + 1) / Double(frame.denoise_steps)
              : 0,
            outputURL: previewURL
          )
        )
      } catch {
        // Keep denoising; the next preview retries the write.
      }
    }
    return isCancelled ? 1 : 0
  }

  /* fileprivate rather than private: the Z-Image lane writes a still
   * without going through the callback context, and one PNG encoder in
   * this file is better than two. */
  fileprivate static func encodePNG(_ frame: h3_frame) -> Data? {
    guard let rgb = frame.rgb, frame.width > 0, frame.height > 0,
      frame.stride >= frame.width * 3,
      let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
    else {
      return nil
    }
    let data = Data(bytes: rgb, count: Int(frame.stride) * Int(frame.height))
    guard let provider = CGDataProvider(data: data as CFData),
      let image = CGImage(
        width: Int(frame.width),
        height: Int(frame.height),
        bitsPerComponent: 8,
        bitsPerPixel: 24,
        bytesPerRow: Int(frame.stride),
        space: colorSpace,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
        provider: provider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
      )
    else {
      return nil
    }
    let output = NSMutableData()
    guard
      let destination = CGImageDestinationCreateWithData(
        output, UTType.png.identifier as CFString, 1, nil
      )
    else {
      return nil
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { return nil }
    return output as Data
  }
}

private enum EngineStillError: Error {
  case missingFrame
}

private func generationProgressCallback(
  _ phase: UnsafePointer<CChar>?,
  _ completed: Int32,
  _ total: Int32,
  _ opaque: UnsafeMutableRawPointer?
) -> Int32 {
  guard let opaque else { return 1 }
  let context = Unmanaged<GenerationCallbackContext>.fromOpaque(opaque).takeUnretainedValue()
  return context.progress(
    phase: phase.map { String(cString: $0) } ?? "Generating",
    completed: completed,
    total: total
  )
}

/// Z-Image reports every stage, not just the sampler: loading and encoding
/// take about two minutes before the first pass, and a run that says nothing
/// until then looks stopped.
///
/// The denoise phase is spelled "denoise step N/M" because that is the form
/// `GenerationRemaining` parses to place the run as a whole. Reporting a bare
/// "denoise" — which this did — left overall progress pinned at zero and the
/// estimate reading "calculating" for the entire generation.
///
/// Returns whether the sampler should keep going, which is the opposite sense
/// to `progress`, whose non-zero means cancel.
private func zimageStepCallback(
  _ phase: UnsafePointer<CChar>?,
  _ completed: Int32,
  _ total: Int32,
  _ opaque: UnsafeMutableRawPointer?
) -> Int32 {
  guard let opaque else { return 1 }
  let context = Unmanaged<GenerationCallbackContext>.fromOpaque(opaque)
    .takeUnretainedValue()
  let name = phase.map { String(cString: $0) } ?? "denoise"
  guard name == "denoise" else {
    return context.progress(phase: name, completed: completed, total: total) == 0
      ? 1 : 0
  }
  // The step counter goes in the phase name, and the fraction reports the
  // step itself as finished. H3 subdivides a pass and reports how far into
  // one it is; a Z-Image pass is atomic, so anything less than whole here
  // would be read as "part way through step N" and place the run at
  // (N-1 + N/M)/M — behind where it actually is, all the way to the end.
  return context.progress(
    phase: "denoise step \(completed)/\(total)", completed: 1, total: 1) == 0
    ? 1 : 0
}

/// LTX reports the same way Z-Image does, with one difference that matters:
/// its tick is asked *between blocks* so a cancel lands inside one rather than
/// at the end of a step, and a step is well over half a minute. So the same
/// step number arrives forty-eight times, and reporting it as a fraction would
/// make the bar stutter backwards. The step counter goes in the phase name and
/// the fraction stays whole, exactly as above.
/// Borrow an array of Swift strings as a C array of C strings for the duration
/// of a call. `withCString` nests one at a time and the engine wants them all
/// at once, so this walks the list recursively rather than nesting by hand.
private func withCStrings<Result>(
  _ strings: [String],
  _ body: (UnsafeBufferPointer<UnsafePointer<CChar>?>) -> Result
) -> Result {
  var pointers: [UnsafePointer<CChar>?] = []
  pointers.reserveCapacity(strings.count)
  func step(_ index: Int) -> Result {
    if index == strings.count {
      return pointers.withUnsafeBufferPointer(body)
    }
    return strings[index].withCString { pointer in
      pointers.append(pointer)
      return step(index + 1)
    }
  }
  return step(0)
}

private func ltxStepCallback(
  _ phase: UnsafePointer<CChar>?,
  _ completed: Int32,
  _ total: Int32,
  _ opaque: UnsafeMutableRawPointer?
) -> Int32 {
  guard let opaque else { return 1 }
  let context = Unmanaged<GenerationCallbackContext>.fromOpaque(opaque)
    .takeUnretainedValue()
  let name = phase.map { String(cString: $0) } ?? "denoise"
  guard name == "denoise" else {
    return context.progress(phase: name, completed: completed, total: total) == 0
      ? 1 : 0
  }
  return context.progress(
    phase: "denoise step \(completed)/\(total)", completed: 1, total: 1) == 0
    ? 1 : 0
}

private func soundEffectStepCallback(
  _ completed: Int32,
  _ total: Int32,
  _ opaque: UnsafeMutableRawPointer?
) {
  guard let opaque else { return }
  let context = Unmanaged<GenerationCallbackContext>.fromOpaque(opaque)
    .takeUnretainedValue()
  _ = context.progress(phase: "denoise", completed: completed, total: total)
}

/// Speech reports one tick per 80 ms frame. `total` is the ceiling the
/// request asked for, not a prediction: the model stops when the line is
/// spoken, which is usually well short of it.
private func speechFrameCallback(
  _ frames: Int32,
  _ total: Int32,
  _ opaque: UnsafeMutableRawPointer?
) {
  guard let opaque else { return }
  let context = Unmanaged<GenerationCallbackContext>.fromOpaque(opaque)
    .takeUnretainedValue()
  _ = context.progress(phase: "speaking", completed: frames, total: total)
}

private func generationFrameCallback(
  _ frame: UnsafePointer<h3_frame>?,
  _ opaque: UnsafeMutableRawPointer?
) -> Int32 {
  guard let opaque else { return 1 }
  let context = Unmanaged<GenerationCallbackContext>.fromOpaque(opaque).takeUnretainedValue()
  guard let frame = frame?.pointee else { return 1 }
  if frame.denoise_step >= 0 {
    return context.deliverPreview(frame)
  }
  if context.capturesStill {
    return context.captureStill(frame)
  }
  return 0
}

/// H3 itself is synchronous. This runner owns the one background generation
/// thread and accepts cancellation from the protocol reader while C is active.
private final class EngineRuntime: @unchecked Sendable {
  static let shared = EngineRuntime()

  private let lock = NSLock()
  private var activeContext: GenerationCallbackContext?

  var isBusy: Bool {
    lock.withLock { activeContext != nil }
  }

  /// Runs `body` only when no generation is active, holding the same lock
  /// `start` uses so a new job cannot begin while the cache is being dropped.
  func performIfIdle(_ body: () -> Void) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    guard activeContext == nil else { return false }
    body()
    return true
  }

  func start(_ command: EngineCommand) {
    guard let request = command.generation, let jobID = command.jobID else {
      EngineOutput.fail(command, message: "Generation parameters and a job ID are required")
      return
    }
    switch request.kind {
    case .audio:
      guard engineCapabilities.supports(.standaloneAudioGeneration) else {
        EngineOutput.fail(command, message: "This engine cannot generate standalone audio")
        return
      }
    case .soundEffect:
      guard engineCapabilities.supports(.soundEffectGeneration) else {
        EngineOutput.fail(command, message: "This engine cannot generate sound effects")
        return
      }
    case .speech:
      guard engineCapabilities.supports(.speechGeneration) else {
        EngineOutput.fail(command, message: "This engine cannot generate speech")
        return
      }
      guard request.speech != nil else {
        EngineOutput.fail(command, message: "Speech needs its voice settings")
        return
      }
    case .video:
      /* Two models render a clip and they are gated separately: an engine may
       * have one package and not the other. */
      if request.video?.model == .ltx {
        guard engineCapabilities.supports(.ltxGeneration) else {
          EngineOutput.fail(command, message: "This engine cannot run LTX-2.5")
          return
        }
        let width = request.canvasWidth ?? 0
        let height = request.canvasHeight ?? 0
        let frames = EngineVideoOptions.frames(forSeconds: request.duration)
        var seconds = 0.0
        var reason = [CChar](repeating: 0, count: 512)
        guard
          h3ddle_ltx_plan(
            Int32(width), Int32(height), Int32(frames),
            Int32(EngineVideoOptions.fps),
            &seconds, &reason, reason.count) != 0
        else {
          EngineOutput.fail(command, message: String(cString: reason))
          return
        }
        /* Conditioning pictures are now encoded and appended to the DiT's
         * sequence, so they are accepted — but there is a ceiling, because
         * every one is a permanent addition every block of every step reads. */
        let conditioning =
          (request.firstFrameURL == nil ? 0 : 1)
          + (request.lastFrameURL == nil ? 0 : 1)
          + request.referenceImageURLs.count
        guard conditioning <= Int(H3DDLE_LTX_MAX_CONDITIONING) else {
          EngineOutput.fail(
            command,
            message: "LTX takes at most \(H3DDLE_LTX_MAX_CONDITIONING) "
              + "conditioning pictures, including start and end frames; "
              + "\(conditioning) were given")
          return
        }
      } else {
        guard engineCapabilities.supports(.videoGeneration) else {
          EngineOutput.fail(command, message: "FFmpeg is required for H3 video output")
          return
        }
      }
    case .image:
      /* Two models can render a still and they are gated separately: an
       * engine may have one package and not the other. */
      if request.image?.model == .zImage {
        guard engineCapabilities.supports(.zImageGeneration) else {
          EngineOutput.fail(command, message: "This engine cannot run Z-Image")
          return
        }
        let width = request.canvasWidth ?? 0
        let height = request.canvasHeight ?? 0
        guard h3ddle_zimage_supports_frame(Int32(width), Int32(height)) != 0
        else {
          EngineOutput.fail(
            command,
            message: "Z-Image cannot render \(width)×\(height); both sides "
              + "must be a multiple of 16 and their token count a multiple "
              + "of 32")
          return
        }
        /* A start frame is the picture to work from and is honoured. The
         * other two still are not: this model conditions on the prompt and
         * on what it is started from, and nothing else, so an end frame or a
         * reference cannot be answered as asked. Refused rather than dropped
         * — a picture that ignores what it was given reads as a bad model
         * rather than a rejected request. */
        guard request.lastFrameURL == nil, request.referenceImageURLs.isEmpty
        else {
          EngineOutput.fail(
            command,
            message: "Z-Image can start from a picture, but it cannot take an "
              + "end frame or reference images")
          return
        }
      } else {
        guard engineCapabilities.supports(.imageGeneration) else {
          EngineOutput.fail(command, message: "This engine cannot produce stills")
          return
        }
      }
    }
    guard let modelDirectory = request.modelDirectory, modelDirectory.isFileURL,
      request.outputURL.isFileURL
    else {
      EngineOutput.fail(command, message: "Local model and output locations are required")
      return
    }

    let callbackContext = GenerationCallbackContext(
      requestID: command.requestID,
      jobID: jobID,
      previewURL: request.outputURL
        .deletingPathExtension()
        .appendingPathExtension("preview.png"),
      capturesStill: request.kind == .image
    )
    let accepted = lock.withLock { () -> Bool in
      guard activeContext == nil else { return false }
      activeContext = callbackContext
      return true
    }
    guard accepted else {
      EngineOutput.fail(command, message: "The H3 engine is already generating")
      return
    }

    EngineOutput.emit(
      EngineEvent(
        requestID: command.requestID,
        jobID: jobID,
        kind: .accepted,
        capabilities: engineCapabilities
      )
    )

    Thread.detachNewThread { [self] in
      run(
        request,
        command: command,
        modelDirectory: modelDirectory,
        callbackContext: callbackContext
      )
    }
  }

  func cancel(_ command: EngineCommand) {
    guard let jobID = command.jobID else {
      EngineOutput.fail(command, message: "A job ID is required for cancellation")
      return
    }
    let context = lock.withLock { activeContext?.jobID == jobID ? activeContext : nil }
    guard let context else {
      EngineOutput.fail(command, message: "The requested generation job is not active")
      return
    }
    context.cancel()
  }

  func cancelActiveJob() {
    lock.withLock { activeContext }?.cancel()
  }

  /// Stable Audio 3 keeps none of H3's machinery: no shared model cache, no
  /// preview decoder, no container to mux. It loads its package, denoises,
  /// and writes a WAV.
  private func runSoundEffect(
    _ request: EngineGenerationRequest,
    command: EngineCommand,
    packageDirectory: URL,
    callbackContext: GenerationCallbackContext
  ) {
    var error = [CChar](repeating: 0, count: 512)
    let opaque = Unmanaged.passUnretained(callbackContext).toOpaque()
    let produced = packageDirectory.path.withCString { packagePath in
      request.prompt.withCString { prompt in
        request.outputURL.path.withCString { outputPath in
          h3ddle_sa3_generate(
            packagePath,
            prompt,
            request.duration,
            Int32(request.denoisingSteps ?? 0),
            request.seed ?? 42,
            outputPath,
            soundEffectStepCallback,
            opaque,
            &error,
            error.count
          )
        }
      }
    }

    if callbackContext.isCancelled {
      EngineOutput.emit(
        EngineEvent(
          requestID: command.requestID,
          jobID: command.jobID,
          kind: .cancelled
        )
      )
      return
    }
    guard produced != 0 else {
      let message = String(cString: error)
      EngineOutput.fail(
        command,
        message: message.isEmpty ? "Sound effect generation failed" : message
      )
      return
    }
    EngineOutput.emit(
      EngineEvent(
        requestID: command.requestID,
        jobID: command.jobID,
        kind: .completed,
        outputURL: request.outputURL,
        outputDuration: request.duration
      )
    )
  }

  /// LTX-2.5: a prompt in, an MP4 with its own soundtrack out.
  ///
  /// Nothing of H3's applies — no shared model cache, no preview decoder, no
  /// reference conditioning — and unlike every other engine here it writes its
  /// own container. A clip is 200 MB of float pixels at 512 square, and there
  /// is nothing this side would do with them but hand them back to the muxer,
  /// so the packing and the write happen in C and `outputURL` receives a
  /// finished file.
  ///
  /// The package is loaded a stage at a time and released as the call returns.
  /// It is not a caching decision: the Gemma tower and the DiT are 37 GB
  /// together and never both resident.
  private func runLTX(
    _ request: EngineGenerationRequest,
    options: EngineVideoOptions,
    command: EngineCommand,
    packageDirectory: URL,
    callbackContext: GenerationCallbackContext
  ) {
    let width = request.canvasWidth ?? 864
    let height = request.canvasHeight ?? 480
    /* Rounded to the nearest renderable length here as well as in the app, so
     * a request that arrived from somewhere else still gets a clip rather than
     * a refusal. */
    let frames = EngineVideoOptions.frames(forSeconds: request.duration)
    var error = [CChar](repeating: 0, count: 512)
    let opaque = Unmanaged.passUnretained(callbackContext).toOpaque()
    /* The conditioning pictures, as C strings that outlive the call. Anchors
     * are named separately from references because they mean different things
     * to the model: an anchor pins an end of the clip, a reference is spread
     * through the middle. */
    let firstPath = request.firstFrameURL?.path ?? ""
    let lastPath = request.lastFrameURL?.path ?? ""
    let referencePaths = request.referenceImageURLs.map(\.path)
    let wrote = packageDirectory.path.withCString { packagePath in
      request.prompt.withCString { prompt in
        "h3_shaders.metal".withCString { shaders in
          request.outputURL.path.withCString { output in
            firstPath.withCString { first in
              lastPath.withCString { last in
                withCStrings(referencePaths) { references in
                  h3ddle_ltx_generate(
                    packagePath,
                    shaders,
                    prompt,
                    Int32(width),
                    Int32(height),
                    Int32(frames),
                    Int32(EngineVideoOptions.fps),
                    Int32(options.steps ?? 0),
                    request.seed ?? 42,
                    firstPath.isEmpty ? nil : first,
                    lastPath.isEmpty ? nil : last,
                    references.baseAddress,
                    Int32(referencePaths.count),
                    output,
                    ltxStepCallback,
                    opaque,
                    &error,
                    error.count
                  )
                }
              }
            }
          }
        }
      }
    }
    guard wrote != 0 else {
      let message = String(cString: error)
      if message.isEmpty {
        /* An empty message with a zero return is the sampler honouring a
         * cancel, not a failure. */
        EngineOutput.emit(
          EngineEvent(
            requestID: command.requestID, jobID: command.jobID, kind: .cancelled))
      } else {
        EngineOutput.fail(command, message: message)
      }
      return
    }
    EngineOutput.emit(
      EngineEvent(
        requestID: command.requestID,
        jobID: command.jobID,
        kind: .completed,
        outputURL: request.outputURL,
        outputDuration: Double(frames) / Double(EngineVideoOptions.fps)
      ))
  }

  /// Z-Image-Turbo, like Stable Audio 3, keeps none of H3's machinery: no
  /// shared model cache, no preview decoder, nothing to mux. It loads its
  /// package, samples, decodes, and writes a PNG.
  ///
  /// The package is released as the call returns rather than cached. Fourteen
  /// gigabytes is not cheap to reload, but the decoder alone wants twelve of
  /// working buffers at the larger canvases, and holding the model resident
  /// would compete with H3 for the memory that actually constrains this app.
  /// A picture at exactly the size the model renders, which is square.
  ///
  /// Anything else is cropped to its middle first rather than squashed: a
  /// portrait squeezed to a landscape comes back as a subtly wrong-shaped
  /// face, which reads as the model being poor rather than the framing being
  /// lost. Cropping is the lesser harm and the one the user can see coming.
  private func framePixels(from url: URL, width: Int, height: Int) -> [UInt8]? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { return nil }
    /* Centre-crop to the *target's* shape rather than to a square: the canvas
     * is no longer always square, and cropping to one would letterbox a
     * landscape source into a landscape frame. */
    let wanted = Double(width) / Double(height)
    var cropWidth = image.width
    var cropHeight = Int((Double(image.width) / wanted).rounded())
    if cropHeight > image.height {
      cropHeight = image.height
      cropWidth = Int((Double(image.height) * wanted).rounded())
    }
    guard cropWidth > 0, cropHeight > 0,
      let cropped = image.cropping(
        to: CGRect(
          x: (image.width - cropWidth) / 2, y: (image.height - cropHeight) / 2,
          width: cropWidth, height: cropHeight))
    else { return nil }

    var pixels = [UInt8](repeating: 0, count: 4 * width * height)
    let drawn: Bool = pixels.withUnsafeMutableBytes { raw in
      guard let context = CGContext(
        data: raw.baseAddress, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: 4 * width,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
      else { return false }
      context.interpolationQuality = .high
      context.draw(cropped, in: CGRect(x: 0, y: 0, width: width, height: height))
      return true
    }
    guard drawn else { return nil }

    /* RGBX down to RGB, which is what the engine takes. */
    var rgb = [UInt8](repeating: 0, count: 3 * width * height)
    for index in 0..<(width * height) {
      rgb[index * 3] = pixels[index * 4]
      rgb[index * 3 + 1] = pixels[index * 4 + 1]
      rgb[index * 3 + 2] = pixels[index * 4 + 2]
    }
    return rgb
  }

  private func runZImage(
    _ request: EngineGenerationRequest,
    options: EngineImageOptions,
    command: EngineCommand,
    packageDirectory: URL,
    callbackContext: GenerationCallbackContext
  ) {
    let width = request.canvasWidth ?? 1024
    let height = request.canvasHeight ?? width
    var error = [CChar](repeating: 0, count: 512)
    var rgb = [UInt8](repeating: 0, count: 3 * width * height)
    let opaque = Unmanaged.passUnretained(callbackContext).toOpaque()

    var source: [UInt8]? = nil
    if let frame = request.firstFrameURL {
      guard let squared = framePixels(from: frame, width: width, height: height)
      else {
        EngineOutput.fail(
          command, message: "That picture cannot be read as an image")
        return
      }
      source = squared
    }
    /* Measured on this checkpoint rather than taken from the usual half-way
     * convention, which is far too low here: at 0.65 an apple asked to become
     * a pear stayed an apple, and at 0.30 a picture came back untouched. The
     * blend region is 0.7 to 0.95, and a default inside it is the difference
     * between a control that appears to work and one that appears ignored. */
    let strength = Float(request.sourceStrength ?? 0.85)
    let produced = packageDirectory.path.withCString { packagePath in
      request.prompt.withCString { prompt in
        "h3_shaders.metal".withCString { shaders in
          rgb.withUnsafeMutableBufferPointer { pixelBuffer in
            (source ?? []).withUnsafeBufferPointer { sourceBuffer in
            let sourcePointer = source == nil ? nil : sourceBuffer.baseAddress
            return h3ddle_zimage_generate(
              packagePath,
              shaders,
              prompt,
              Int32(width),
              Int32(height),
              Int32(options.steps ?? 0),
              request.seed ?? 42,
              sourcePointer,
              source == nil ? 1.0 : strength,
              pixelBuffer.baseAddress,
              zimageStepCallback,
              opaque,
              &error,
              error.count
            )
            }
          }
        }
      }
    }
    guard produced != 0 else {
      let message = String(cString: error)
      if message.isEmpty {
        /* An empty message with a zero return is the sampler honouring a
         * cancel, not a failure. */
        EngineOutput.emit(
          EngineEvent(
            requestID: command.requestID, jobID: command.jobID, kind: .cancelled))
      } else {
        EngineOutput.fail(command, message: message)
      }
      return
    }
    var frame = h3_frame()
    frame.width = Int32(width)
    frame.height = Int32(height)
    frame.stride = Int32(width * 3)
    let wrote: Bool = rgb.withUnsafeBufferPointer { pixelBuffer in
      frame.rgb = pixelBuffer.baseAddress
      guard let png = GenerationCallbackContext.encodePNG(frame) else {
        return false
      }
      return (try? png.write(to: request.outputURL, options: .atomic)) != nil
    }
    guard wrote else {
      EngineOutput.fail(command, message: "Could not write the picture")
      return
    }
    EngineOutput.emit(
      EngineEvent(
        requestID: command.requestID,
        jobID: command.jobID,
        kind: .completed,
        outputURL: request.outputURL
      )
    )
  }

  /// Qwen3-TTS, like Stable Audio 3, keeps none of H3's machinery. The
  /// duration is a ceiling rather than a target — the model stops when the
  /// line is spoken — so the completed event reports what was produced.
  private func runSpeech(
    _ request: EngineGenerationRequest,
    speech: EngineSpeechOptions,
    command: EngineCommand,
    packageDirectory: URL,
    callbackContext: GenerationCallbackContext
  ) {
    var error = [CChar](repeating: 0, count: 512)
    var produced = 0.0
    let opaque = Unmanaged.passUnretained(callbackContext).toOpaque()
    // Exactly one of these carries the voice, and either may be absent: with
    // neither, the model speaks unconditioned.
    let referencePath = speech.referenceAudioURL?.path ?? ""
    let embeddingPath = speech.voiceEmbeddingURL?.path ?? ""
    let wrote = packageDirectory.path.withCString { packagePath in
      request.prompt.withCString { text in
        speech.language.rawValue.withCString { language in
          referencePath.withCString { reference in
            embeddingPath.withCString { embedding in
            request.outputURL.path.withCString { outputPath in
              h3ddle_qwen_generate(
                packagePath,
                text,
                language,
                reference,
                embedding,
                request.duration,
                speech.temperature,
                Int32(speech.topK),
                speech.repetitionPenalty,
                request.seed ?? 42,
                outputPath,
                speechFrameCallback,
                opaque,
                &produced,
                &error,
                error.count
              )
            }
            }
          }
        }
      }
    }

    if callbackContext.isCancelled {
      EngineOutput.emit(
        EngineEvent(
          requestID: command.requestID,
          jobID: command.jobID,
          kind: .cancelled
        )
      )
      return
    }
    guard wrote != 0 else {
      let message = String(cString: error)
      EngineOutput.fail(
        command,
        message: message.isEmpty ? "Speech generation failed" : message
      )
      return
    }
    EngineOutput.emit(
      EngineEvent(
        requestID: command.requestID,
        jobID: command.jobID,
        kind: .completed,
        outputURL: request.outputURL,
        outputDuration: produced
      )
    )
  }

  private func run(
    _ request: EngineGenerationRequest,
    command: EngineCommand,
    modelDirectory: URL,
    callbackContext: GenerationCallbackContext
  ) {
    defer {
      lock.withLock {
        if activeContext === callbackContext {
          activeContext = nil
        }
      }
      EngineResourceWatch.shared.noteActivity()
    }

    if request.kind == .soundEffect {
      runSoundEffect(
        request,
        command: command,
        packageDirectory: modelDirectory,
        callbackContext: callbackContext
      )
      return
    }

    if request.kind == .video, request.video?.model == .ltx {
      runLTX(
        request,
        options: request.video ?? EngineVideoOptions(model: .ltx),
        command: command,
        packageDirectory: modelDirectory,
        callbackContext: callbackContext
      )
      return
    }

    if request.kind == .image, request.image?.model == .zImage {
      runZImage(
        request,
        options: request.image ?? EngineImageOptions(model: .zImage),
        command: command,
        packageDirectory: modelDirectory,
        callbackContext: callbackContext
      )
      return
    }

    if request.kind == .speech, let speech = request.speech {
      runSpeech(
        request,
        speech: speech,
        command: command,
        packageDirectory: modelDirectory,
        callbackContext: callbackContext
      )
      return
    }

    modelDirectory.path.withCString { modelPath in
      guard let context = EngineModelStore.shared.load(path: modelPath) else {
        EngineOutput.fail(command, message: lastH3Error(nil))
        return
      }

      var parameters = h3ddle_h3_default_params()
      // Community stills decode one frame of a short H3 clip. The full
      // 22-frame chunk is the default because its extra latent time steps
      // resolve visibly more detail; the 5-frame first chunk is the same
      // trained shape at about a third of the cost, offered as a choice.
      // Community audio is the same joint model rendered for its soundtrack.
      let stillRequested = request.kind == .image
      let audioRequested = request.kind == .audio
      parameters.frames =
        stillRequested
        ? (request.fastStill ? 5 : 22)
        : h3ddle_h3_frames_for_seconds(request.duration)
      // A still keeps one frame, so decode one instead of the whole clip.
      parameters.still_frame_only = stillRequested ? 1 : 0
      // An audio job keeps no pictures at all, so skip the video decoder and
      // let the engine write the soundtrack straight out as a WAV.
      parameters.audio_only = audioRequested ? 1 : 0
      parameters.preview_denoise = request.previewDenoise && !audioRequested ? 1 : 0
      parameters.on_frame =
        (request.previewDenoise && !audioRequested) || stillRequested
        ? generationFrameCallback : nil
      parameters.on_progress = generationProgressCallback
      // Every request carries a validated preset; without this the engine
      // falls back to its 864x480 close-reference defaults regardless of
      // model format. Native 256 square with four passes remains the floor:
      // the earlier 64-square/two-pass plumbing smoke produced block noise
      // and must never be exposed as a generation preset.
      // h3.c halves spatial RoPE coordinates at exactly 256x256, a heuristic
      // validated on the released BF16 weights only. On the pruned INT8
      // package it swaps prompt subjects (a sitting cat rendered as a man at
      // a table; reproduced 2026-08-13 with seed 42, fixed by
      // --use-reference-rope), so restore reference coordinates there.
      if let model = h3_model(context),
        modelFormat(model.pointee) == .optimizedINT8SingleFile
      {
        parameters.use_reference_rope = 1
      }
      let quality = request.quality
      if audioRequested {
        // The soundtrack is generated jointly with pictures nobody keeps, so
        // the canvas is kept at the mechanical minimum.
        //
        // KNOWN ISSUE: prompts describing ambience (rain, thunder) come back as
        // speech. Raising the canvas does not fix it — 32, 64, 128, and 256 were
        // compared on one prompt at a fixed seed and pass count on 2026-08-14
        // and every one returned speech, so 256 costs about six times per pass
        // (148s against 24.7s for a three-second clip) and buys nothing. The
        // cause is still unknown; the canvas is not it.
        let audioCanvas = Int32(EngineGenerationRequest.audioCanvasSize)
        parameters.width = audioCanvas
        parameters.height = audioCanvas
      } else if let width = request.canvasWidth, let height = request.canvasHeight {
        parameters.width = Int32(width)
        parameters.height = Int32(height)
      } else {
        let canvas = Int32(quality.canvasSize)
        parameters.width = canvas
        parameters.height = canvas
      }
      parameters.render_width = 0
      parameters.render_height = 0
      parameters.steps = Int32(quality.denoisingSteps)
      parameters.dit_layers = Int32(quality.activeDiTLayers)
      parameters.denoise_reuse = Int32(quality.denoiseReuse)
      if let steps = request.denoisingSteps {
        parameters.steps = Int32(steps)
        // Whole-denoiser reuse needs a real budget: below ten passes every
        // requested pass has to run the model or quality collapses.
        if steps < 10 {
          parameters.denoise_reuse = 1
        }
      }
      if let layers = request.activeDiTLayers {
        parameters.dit_layers = Int32(layers)
      }
      if let coreReuse = request.coreReuse, coreReuse > 1 {
        // The engine rejects core reuse combined with whole-denoiser reuse.
        parameters.core_reuse = Int32(coreReuse)
        parameters.denoise_reuse = 1
      }
      if request.blockCache {
        // The block cache replaces both reuse ladders; the engine refuses
        // the combinations outright, so resolve them here.
        parameters.block_cache = 1
        parameters.core_reuse = 1
        parameters.denoise_reuse = 1
      }
      parameters.use_beta_schedule = request.useBetaSchedule ? 1 : 0
      if let seed = request.seed {
        parameters.seed = seed
      }

      let hasFrames = request.firstFrameURL != nil || request.lastFrameURL != nil
      let references = Array(
        request.referenceImageURLs.prefix(EngineGenerationRequest.referenceImageLimit))
      if hasFrames, !references.isEmpty {
        EngineOutput.fail(
          command,
          message: "Start/end frames cannot be combined with reference images."
        )
        return
      }

      let firstPath = request.firstFrameURL?.path
      let lastPath = request.lastFrameURL?.path
      let referencePaths = references.map(\.path)

      let opaque = Unmanaged.passRetained(callbackContext).toOpaque()
      parameters.callback_opaque = opaque
      defer { Unmanaged<GenerationCallbackContext>.fromOpaque(opaque).release() }

      withOptionalCString(firstPath) { firstC in
        withOptionalCString(lastPath) { lastC in
          withCStringList(referencePaths) { referenceCs in
            parameters.first_frame = firstC
            parameters.last_frame = lastC
            let referenceRecords = referenceCs.map { path in
              h3_reference(
                kind: H3_REFERENCE_IMAGE,
                path: path,
                audio_path: nil,
                include_embedded_audio: 0
              )
            }
            referenceRecords.withUnsafeBufferPointer { buffer in
              if !buffer.isEmpty {
                parameters.references = buffer.baseAddress
                parameters.reference_count = buffer.count
              }
      request.prompt.withCString { prompt in
        request.outputURL.path.withCString { outputPath in
          // Stills skip the container and keep a PNG. Audio writes a WAV.
          parameters.output_path = stillRequested ? nil : outputPath
          guard let result = h3_generate(context, prompt, &parameters) else {
            if callbackContext.isCancelled {
              EngineOutput.emit(
                EngineEvent(
                  requestID: command.requestID,
                  jobID: command.jobID,
                  kind: .cancelled
                )
              )
            } else {
              EngineOutput.fail(command, message: lastH3Error(context))
            }
            return
          }
          let muxedDuration =
            result.pointee.fps > 0
            ? Double(result.pointee.frames) / Double(result.pointee.fps)
            : request.duration
          h3_result_free(result)

          if callbackContext.isCancelled {
            EngineOutput.emit(
              EngineEvent(
                requestID: command.requestID,
                jobID: command.jobID,
                kind: .cancelled
              )
            )
            return
          }

          if stillRequested {
            do {
              try callbackContext.writeStill(to: request.outputURL)
            } catch {
              EngineOutput.fail(
                command,
                message: "H3 finished without a decodable still frame"
              )
              return
            }
          }

          EngineOutput.emit(
            EngineEvent(
              requestID: command.requestID,
              jobID: command.jobID,
              kind: .completed,
              outputURL: request.outputURL,
              outputDuration: stillRequested ? request.duration : muxedDuration
            )
          )
        }
      }
            }
          }
        }
      }
    }
  }
}
