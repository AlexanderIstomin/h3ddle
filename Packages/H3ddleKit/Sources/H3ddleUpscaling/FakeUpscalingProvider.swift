import Foundation
import H3ddleCore

public struct FakeUpscalingProvider: UpscalingProvider {
  private let stepDelay: Duration

  public init(stepDelay: Duration = .seconds(1)) {
    self.stepDelay = stepDelay
  }

  public func capabilities(for request: UpscalingRequest) async -> [UpscalingCapability] {
    guard request.sourceKind == .image || request.sourceKind == .video else { return [] }
    return [
      UpscalingCapability(
        backend: .fake,
        displayName: "Fake upscaler",
        supportedMediaKinds: [.image, .video],
        supportsArbitraryOutputSize: true,
        usesTemporalFrames: request.sourceKind == .video,
        availability: .ready
      )
    ]
  }

  public func events(
    for request: UpscalingRequest
  ) -> AsyncThrowingStream<UpscalingEvent, any Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          try request.validate()
          continuation.yield(.preparing(backend: .fake))

          let phases = ["Reading source", "Upscaling", "Writing asset"]
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

          try FileManager.default.createDirectory(
            at: request.destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
          )
          if FileManager.default.fileExists(atPath: request.sourceURL.path) {
            let data = try Data(contentsOf: request.sourceURL)
            try data.write(to: request.destinationURL, options: .atomic)
          } else {
            try Data().write(to: request.destinationURL, options: .atomic)
          }

          continuation.yield(
            .completed(
              UpscalingResult(
                requestID: request.id,
                outputURL: request.destinationURL,
                mediaKind: request.sourceKind,
                pixelSize: request.targetPixelSize,
                duration: request.sourceDuration
              )
            )
          )
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
