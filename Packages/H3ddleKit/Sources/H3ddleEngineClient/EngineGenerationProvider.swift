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

  public func events(
    for request: GenerationRequest
  ) -> AsyncThrowingStream<GenerationEvent, any Error> {
    AsyncThrowingStream { continuation in
      guard request.kind == .video || request.kind == .image else {
        continuation.finish(
          throwing: EngineGenerationProviderError.unsupportedKind(request.kind)
        )
        return
      }
      guard !request.prompt.isEmpty else {
        continuation.finish(throwing: GenerationError.emptyPrompt)
        return
      }

      let jobID = UUID()
      let outputURL =
        outputDirectory
        .appendingPathComponent(jobID.uuidString)
        .appendingPathExtension(request.kind == .image ? "png" : "mp4")
      let command = EngineCommand(
        jobID: jobID,
        kind: .generate,
        generation: EngineGenerationRequest(
          kind: request.kind == .image ? .image : .video,
          prompt: request.prompt,
          duration: request.duration,
          quality: request.quality,
          denoisingSteps: request.denoisingSteps,
          activeDiTLayers: request.activeDiTLayers,
          coreReuse: request.coreReuse,
          previewDenoise: request.previewDenoise,
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
              continuation.yield(
                .completed(
                  AssetReference(
                    kind: request.kind.mediaKind,
                    displayName: request.kind == .image
                      ? "Generated H3 Image"
                      : "Generated H3 Video",
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
            case .ready, .modelInspected, .accepted:
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
