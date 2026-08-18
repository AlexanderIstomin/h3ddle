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
  static func outputExtension(for kind: GenerationKind) -> String {
    switch kind {
    case .image: "png"
    case .audio: "wav"
    case .video: "mp4"
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

      let jobID = UUID()
      let outputURL =
        outputDirectory
        .appendingPathComponent(jobID.uuidString)
        .appendingPathExtension(outputExtension)
      let command = EngineCommand(
        jobID: jobID,
        kind: .generate,
        generation: EngineGenerationRequest(
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
          video: request.kind == .video && request.videoEngine == .ltx
            ? EngineVideoOptions(model: .ltx, steps: request.denoisingSteps)
            : nil,
          modelDirectory: modelDirectory,
          outputURL: outputURL
        )
      )
      let session = session
      let task = Task.detached(priority: .userInitiated) {
        do {
          try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
          )
          try session.run(command: command) { event in
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
                continuation.finish(throwing: EngineGenerationProviderError.missingOutput)
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
        } catch is CancellationError {
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }

      continuation.onTermination = { _ in
        session.cancel(jobID: jobID)
        task.cancel()
      }
    }
  }
}
