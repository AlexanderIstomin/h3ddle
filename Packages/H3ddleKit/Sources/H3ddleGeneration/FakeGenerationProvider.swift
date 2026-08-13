import Foundation
import H3ddleCore

public struct FakeGenerationProvider: GenerationProvider {
  private let stepDelay: Duration

  public init(stepDelay: Duration = .seconds(1)) {
    self.stepDelay = stepDelay
  }

  public func events(
    for request: GenerationRequest
  ) -> AsyncThrowingStream<GenerationEvent, any Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          guard !request.prompt.isEmpty else {
            throw GenerationError.emptyPrompt
          }

          let phases = ["Preparing", "Generating", "Finalizing"]
          for (index, phase) in phases.enumerated() {
            try await Task.sleep(for: stepDelay)
            try Task.checkCancellation()
            continuation.yield(
              .progress(
                phase: phase,
                fractionComplete: Double(index + 1) / Double(phases.count)
              )
            )
          }

          let fileExtension: String =
            switch request.kind {
            case .video: "mp4"
            case .image: "png"
            case .audio: "m4a"
            }
          let asset = AssetReference(
            kind: request.kind.mediaKind,
            displayName: "Generated \(request.kind.rawValue.capitalized)",
            url: FileManager.default.temporaryDirectory
              .appendingPathComponent("h3ddle-fake-\(UUID().uuidString)")
              .appendingPathExtension(fileExtension),
            duration: request.kind == .image ? max(request.duration, 3) : request.duration
          )
          continuation.yield(.completed(asset))
          continuation.finish()
        } catch is CancellationError {
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }

      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }
}
