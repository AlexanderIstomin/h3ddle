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
    var features: [EngineFeature] = [
      .modelInspection, .imageGeneration, .embeddedAudio, .cancellation,
      .denoisingPreviews,
    ]
    if executableIsAvailable(named: "ffmpeg", override: "H3_FFMPEG") {
      features.append(.videoGeneration)
    }
    return features
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

  func load(path: UnsafePointer<CChar>) -> OpaquePointer? {
    let directoryPath = String(cString: path)
    lock.lock()
    defer { lock.unlock() }
    if let context, directory == directoryPath {
      return context
    }
    releaseLocked()
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
    }
  }
}

private func executableIsAvailable(named name: String, override key: String) -> Bool {
  let environment = ProcessInfo.processInfo.environment
  if let override = environment[key], FileManager.default.isExecutableFile(atPath: override) {
    return true
  }
  return (environment["PATH"] ?? "")
    .split(separator: ":")
    .map { URL(fileURLWithPath: String($0)).appendingPathComponent(name).path }
    .contains(where: FileManager.default.isExecutableFile(atPath:))
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

  private static func encodePNG(_ frame: h3_frame) -> Data? {
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
      EngineOutput.fail(command, message: "h3.c does not support standalone audio generation")
      return
    case .video:
      guard engineCapabilities.supports(.videoGeneration) else {
        EngineOutput.fail(command, message: "FFmpeg is required for H3 video output")
        return
      }
    case .image:
      guard engineCapabilities.supports(.imageGeneration) else {
        EngineOutput.fail(command, message: "This engine cannot produce stills")
        return
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

    modelDirectory.path.withCString { modelPath in
      guard let context = EngineModelStore.shared.load(path: modelPath) else {
        EngineOutput.fail(command, message: lastH3Error(nil))
        return
      }

      var parameters = h3ddle_h3_default_params()
      // Community stills are a short H3 clip plus one decoded frame. h3.c
      // rejects anything below one trained 22-frame VAE chunk.
      let stillRequested = request.kind == .image
      parameters.frames =
        stillRequested ? 22 : h3ddle_h3_frames_for_seconds(request.duration)
      parameters.preview_denoise = request.previewDenoise ? 1 : 0
      parameters.on_frame =
        request.previewDenoise || stillRequested ? generationFrameCallback : nil
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
      parameters.width = Int32(quality.canvasSize)
      parameters.height = Int32(quality.canvasSize)
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

      let opaque = Unmanaged.passRetained(callbackContext).toOpaque()
      parameters.callback_opaque = opaque
      defer { Unmanaged<GenerationCallbackContext>.fromOpaque(opaque).release() }

      request.prompt.withCString { prompt in
        request.outputURL.path.withCString { outputPath in
          // Skip FFmpeg mux for stills; the last decoded frame is written as PNG.
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
