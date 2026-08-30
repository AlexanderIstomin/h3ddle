import CryptoKit
import Foundation
import H3ddleCore
import H3ddleEngineProtocol
import H3ddleGeneration

public enum EngineGenerationProviderError: LocalizedError, Equatable, Sendable {
  case unsupportedKind(GenerationKind)
  case missingOutput
  case engineStopped(String)

  public var errorDescription: String? {
    switch self {
    case .unsupportedKind(let kind):
      "The H3 engine cannot generate standalone \(kind.rawValue) output."
    case .missingOutput:
      "The H3 engine completed without an output file."
    case .engineStopped(let message):
      message
    }
  }
}

public struct EngineGenerationProvider: GenerationProvider, Sendable {
  public var session: EngineSession
  public var modelDirectory: URL
  public var outputDirectory: URL

  public init(
    session: EngineSession,
    modelDirectory: URL,
    outputDirectory: URL = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleGenerated", isDirectory: true)
  ) {
    self.session = session
    self.modelDirectory = modelDirectory
    self.outputDirectory = outputDirectory
  }

  public init(
    executableURL: URL,
    modelDirectory: URL,
    outputDirectory: URL = FileManager.default.temporaryDirectory
      .appendingPathComponent("H3ddleGenerated", isDirectory: true)
  ) {
    self.init(
      session: EngineSession(executableURL: executableURL),
      modelDirectory: modelDirectory,
      outputDirectory: outputDirectory
    )
  }

  /// Container the helper writes for each kind. Audio is the audio VAE's own
  /// samples written straight out, not a lossy mux, so it carries no video
  /// and needs no FFmpeg.
  public static func outputExtension(for kind: GenerationKind) -> String {
    switch kind {
    case .image: "png"
    case .audio: "wav"
    case .video: "mp4"
    }
  }

  static func checkpointFingerprint(
    for request: EngineGenerationRequest,
    environment: [String: String] = ProcessInfo.processInfo.environment
  ) throws -> String {
    var canonical = request
    canonical.outputURL = URL(fileURLWithPath: "/H3ddleGenerationOutput")
    canonical.checkpoint = nil
    var hasher = SHA256()
    hasher.update(data: try EngineLineCodec.encode(canonical))
    let process = ProcessInfo.processInfo
    updateFingerprint(
      "protocol:\(H3ddleEngineProtocol.currentVersion)", hasher: &hasher)
    updateFingerprint(
      "host:\(process.operatingSystemVersionString):\(process.processorCount):"
        + "\(process.physicalMemory)",
      hasher: &hasher
    )
    for key in ["CFBundleShortVersionString", "CFBundleVersion"] {
      if let version = Bundle.main.object(forInfoDictionaryKey: key) as? String {
        updateFingerprint("\(key):\(version)", hasher: &hasher)
      }
    }

    let h3Environment = environment
      .filter { $0.key.hasPrefix("H3_") }
      .sorted { $0.key < $1.key }
    for (key, value) in h3Environment {
      updateFingerprint("env:\(key)=\(value)", hasher: &hasher)
      if key.hasSuffix("_PATH"), !value.isEmpty {
        try updateFileIdentity(URL(fileURLWithPath: value), hasher: &hasher)
      }
    }

    var inputs = [request.modelDirectory, request.firstFrameURL, request.lastFrameURL]
      .compactMap { $0 }
    inputs.append(contentsOf: request.referenceImageURLs)
    if let inpainting = request.video?.inpainting {
      inputs.append(inpainting.sourceVideoURL)
      inputs.append(inpainting.maskURL)
    }
    for input in inputs {
      try updateFileIdentity(input, hasher: &hasher)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }

  private static func updateFingerprint(
    _ value: String,
    hasher: inout SHA256
  ) {
    let data = Data(value.utf8)
    var length = UInt64(data.count).littleEndian
    withUnsafeBytes(of: &length) { hasher.update(data: Data($0)) }
    hasher.update(data: data)
  }

  private static func updateFileIdentity(
    _ url: URL,
    hasher: inout SHA256
  ) throws {
    let fileManager = FileManager.default
    let standardized = url.standardizedFileURL
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(
      atPath: standardized.path, isDirectory: &isDirectory)
    else {
      updateFingerprint("missing:\(standardized.path)", hasher: &hasher)
      return
    }
    let keys: Set<URLResourceKey> = [
      .isRegularFileKey, .fileSizeKey, .contentModificationDateKey,
    ]
    let urls: [URL]
    if isDirectory.boolValue {
      guard let enumerator = fileManager.enumerator(
        at: standardized,
        includingPropertiesForKeys: Array(keys),
        options: [.skipsHiddenFiles]
      ) else {
        throw CocoaError(.fileReadUnknown)
      }
      urls = enumerator.compactMap { $0 as? URL }.sorted { $0.path < $1.path }
    } else {
      urls = [standardized]
    }
    updateFingerprint("root:\(standardized.path)", hasher: &hasher)
    for file in urls {
      let values = try file.resourceValues(forKeys: keys)
      guard values.isRegularFile == true else { continue }
      let relative = file.path.dropFirst(standardized.path.count)
      let modified = values.contentModificationDate?.timeIntervalSince1970 ?? 0
      updateFingerprint(
        "file:\(relative):\(values.fileSize ?? -1):\(modified)",
        hasher: &hasher
      )
    }
  }

  public func events(
    for request: GenerationRequest
  ) -> AsyncThrowingStream<GenerationEvent, any Error> {
    AsyncThrowingStream { continuation in
      let outputExtension = Self.outputExtension(for: request.kind)
      let engineKind: EngineGenerationKind =
        switch request.kind {
        case .image: .image
        case .audio:
          switch request.audioEngine {
          case .h3: .audio
          case .stableAudio: .soundEffect
          case .speech: .speech
          }
        case .video: .video
        }
      guard !request.prompt.isEmpty else {
        continuation.finish(throwing: GenerationError.emptyPrompt)
        return
      }

      let jobID = request.recovery?.jobID ?? UUID()
      let outputURL = request.recovery?.outputURL ??
        outputDirectory
          .appendingPathComponent(jobID.uuidString)
          .appendingPathExtension(outputExtension)
      var engineRequest = EngineGenerationRequest(
          kind: engineKind,
          prompt: request.prompt,
          duration: request.duration,
          quality: request.quality,
          denoisingSteps: request.denoisingSteps,
          activeDiTLayers: request.activeDiTLayers,
          coreReuse: request.coreReuse,
          fastStill: request.fastStill,
          blockCache: request.blockCache,
          previewDenoise: request.previewDenoise,
          useBetaSchedule: request.useBetaSchedule,
          h3ModelProfile: request.h3ModelProfile ?? .standard,
          seed: request.seed,
          sourceStrength: request.sourceStrength,
          canvasWidth: request.canvasWidth,
          canvasHeight: request.canvasHeight,
          firstFrameURL: request.firstFrameURL,
          lastFrameURL: request.lastFrameURL,
          referenceImageURLs: request.referenceImageURLs,
          speech: request.speech,
          image: request.kind == .image && request.imageEngine == .zImage
            ? EngineImageOptions(model: .zImage, steps: request.denoisingSteps)
            : nil,
          video: request.kind == .video
            ? (request.videoEngine == .ltx
              ? EngineVideoOptions(
                model: .ltx,
                steps: request.denoisingSteps,
                inpainting: request.videoInpainting
              )
              : request.videoInpainting.map {
                EngineVideoOptions(
                  model: .h3,
                  steps: request.denoisingSteps,
                  inpainting: $0
                )
              })
            : nil,
          allowsLTXMemoryOvercommit: request.allowsLTXMemoryOvercommit == true,
          modelDirectory: modelDirectory,
          outputURL: outputURL
      )
      if let recovery = request.recovery {
        do {
          let fingerprint = try Self.checkpointFingerprint(for: engineRequest)
          engineRequest.checkpoint = EngineCheckpointOptions(
            fileURL: recovery.directoryURL.appendingPathComponent(
              EngineGenerationRecoveryStore.checkpointName),
            fingerprint: fingerprint
          )
        } catch {
          continuation.finish(throwing: error)
          return
        }
      }
      let command = EngineCommand(
        jobID: jobID,
        kind: .generate,
        generation: engineRequest
      )
      let session = session
      let task = Task.detached(priority: .userInitiated) {
        do {
          try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
          )
          if let recovery = request.recovery {
            try FileManager.default.createDirectory(
              at: recovery.directoryURL,
              withIntermediateDirectories: true,
              attributes: [.posixPermissions: 0o700]
            )
          }
          var restartAttempts = 0
          while true {
            do {
              try session.run(command: command) { event in
                if let performance = event.performance {
                  continuation.yield(
                    .resourceUsage(
                      GenerationResourceUsage(
                        physicalFootprintBytes: performance.physicalFootprintBytes
                      )
                    )
                  )
                }
                switch event.kind {
                case .progress:
                  continuation.yield(
                    .progress(
                      phase: event.phase ?? "Generating",
                      fractionComplete: event.fractionComplete ?? 0
                    )
                  )
                  return false
                case .preview:
                  if let previewURL = event.outputURL {
                    continuation.yield(.preview(previewURL))
                  }
                  return false
                case .completed:
                  guard let completedURL = event.outputURL else {
                    continuation.finish(
                      throwing: EngineGenerationProviderError.missingOutput)
                    return true
                  }
                  let displayName =
                    switch request.kind {
                    case .image: "Generated H3 Image"
                    case .audio: "Generated H3 Audio"
                    case .video: "Generated H3 Video"
                    }
                  continuation.yield(
                    .completed(
                      AssetReference(
                        kind: request.kind.mediaKind,
                        displayName: displayName,
                        url: completedURL,
                        duration: event.outputDuration ?? request.duration
                      )
                    )
                  )
                  if let recovery = request.recovery {
                    EngineGenerationRecoveryStore().discard(recovery)
                  }
                  continuation.finish()
                  return true
                case .cancelled:
                  continuation.finish()
                  return true
                case .failed:
                  continuation.finish(
                    throwing: EngineGenerationProviderError.engineStopped(
                      event.message ?? "The H3 engine stopped generation."
                    )
                  )
                  return true
                case .ready, .modelInspected, .accepted, .residency:
                  return false
                }
              }
              break
            } catch let error as EngineGenerationProviderError {
              guard request.recovery != nil, restartAttempts == 0,
                case .engineStopped = error
              else { throw error }
              restartAttempts += 1
              continuation.yield(
                .progress(
                  phase: "Recovering interrupted generation",
                  fractionComplete: 0
                )
              )
            }
          }
        } catch is CancellationError {
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }

      continuation.onTermination = { termination in
        // `finish()` also invokes this callback. Cancelling a job that just
        // completed can race the queue's next command and make the session
        // preempt (kill) a healthy helper between two sequential jobs.
        if case .cancelled = termination {
          session.cancel(jobID: jobID)
        }
        task.cancel()
      }
    }
  }
}
